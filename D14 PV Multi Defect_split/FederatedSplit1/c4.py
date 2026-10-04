import socket
import pickle
import torch
import os
import shutil
import random
from ultralytics import YOLO
import yaml
import json
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import traceback
import sys
import time
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Modules
class ECA(nn.Module):
    """Efficient Channel Attention Module"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
        
        # Adaptive kernel size
        k_size = int(abs((np.log2(channels) + self.b) / self.gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class GhostConv(nn.Module):
    """Ghost Convolution as in GhostNet"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2):
        super(GhostConv, self).__init__()
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

class MobileNetV3Block(nn.Module):
    """MobileNetV3 Block with Squeeze-and-Excitation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 use_se=True, activation='hswish'):
        super(MobileNetV3Block, self).__init__()
        
        self.use_se = use_se
        self.stride = stride
        
        # Expansion phase
        expanded_channels = in_channels * 6
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        self.expand_act = nn.Hardswish() if activation == 'hswish' else nn.ReLU()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, 
                                       kernel_size, stride, kernel_size//2, 
                                       groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_act = nn.Hardswish() if activation == 'hswish' else nn.ReLU()
        
        # Squeeze-and-Excitation
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, expanded_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(expanded_channels // 4, expanded_channels, 1),
                nn.Hardsigmoid() if activation == 'hswish' else nn.Sigmoid()
            )
        
        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        out = self.expand_act(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        out = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(out)))
        
        # SE
        if self.use_se:
            out = out * self.se(out)
        
        # Projection
        out = self.project_bn(self.project_conv(out))
        
        # Skip connection
        if self.stride == 1 and identity.shape == out.shape:
            out = out + identity
            
        return out

class LateralGhostConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LateralGhostConv, self).__init__()
        self.ghost_conv = GhostConv(in_channels, out_channels)
        
    def forward(self, x):
        return self.ghost_conv(x)

class LearnableScalarFusion(nn.Module):
    """Learnable Scalar Fusion to replace ASFF"""
    def __init__(self, num_levels=3):
        super(LearnableScalarFusion, self).__init__()
        self.num_levels = num_levels
        # Learnable scalar weights for each level
        self.weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, features):
        # features: list of feature maps [p3, p4, p5]
        target_size = features[0].size()[2:]  # Use p3 as target
        
        # Resize all features to target level
        resized = []
        for i, feat in enumerate(features):
            if i == 0:
                resized.append(feat)
            else:
                # Downsample to p3 size
                scale_factor = 2 ** i
                feat_resized = F.interpolate(feat, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
                resized.append(feat_resized)
        
        # Apply learnable weights
        weights = self.softmax(self.weights)
        
        # Weighted sum
        out = 0
        for i in range(self.num_levels):
            out += weights[i] * resized[i]
            
        return out

class BiFPN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN_Block, self).__init__()
        self.conv = GhostConv(in_channels, out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.conv(x))

class BiFPN_ECA(nn.Module):
    """BiFPN with ECA attention"""
    def __init__(self, channels=128):
        super(BiFPN_ECA, self).__init__()
        # Feature scaling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Top-down pathway with ECA
        self.td_conv1 = BiFPN_Block(channels, channels)
        self.td_eca1 = ECA(channels)
        self.td_c3_1 = C3(channels, channels, n=2)
        
        self.td_conv2 = BiFPN_Block(channels, channels)
        self.td_eca2 = ECA(channels)
        self.td_c3_2 = C3(channels, channels, n=2)
        
        # P3 refinement
        self.td_conv3 = BiFPN_Block(channels, channels)
        self.td_eca3 = ECA(channels)
        self.td_c3_3 = C3(channels, channels, n=2)
        
        # Bottom-up pathway with ECA
        self.bu_conv1 = BiFPN_Block(channels, channels)
        self.bu_eca1 = ECA(channels)
        self.bu_c3_1 = C3(channels, channels, n=2)
        
        self.bu_conv2 = BiFPN_Block(channels, channels)
        self.bu_eca2 = ECA(channels)
        self.bu_c3_2 = C3(channels, channels, n=2)
        
        # P5 refinement
        self.bu_conv3 = BiFPN_Block(channels, channels)
        self.bu_eca3 = ECA(channels)
        self.bu_c3_3 = C3(channels, channels, n=2)
        
    def forward(self, inputs):
        # Unpack inputs (P3, P4, P5)
        p3, p4, p5 = inputs
        
        # ========== Top-down pathway ==========
        # Level P5 (level 5)
        td5 = p5
        td5 = self.td_eca1(self.td_conv1(td5))
        td5 = self.td_c3_1(td5)
        
        # Level P5 to P4
        td4 = p4 + self.upsample(td5)
        td4 = self.td_eca2(self.td_conv2(td4))
        td4 = self.td_c3_2(td4)
        
        # Level P4 to P3
        td3 = p3 + self.upsample(td4)
        # P3 refinement
        td3 = self.td_eca3(self.td_conv3(td3))
        td3 = self.td_c3_3(td3)
        
        # ========== Bottom-up pathway ==========
        # Level P3 (level 3)
        bu3 = td3
        bu3 = self.bu_eca1(self.bu_conv1(bu3))
        bu3 = self.bu_c3_1(bu3)
        
        # Level P3 to P4
        bu4 = td4 + self.downsample(bu3)
        bu4 = self.bu_eca2(self.bu_conv2(bu4))
        bu4 = self.bu_c3_2(bu4)
        
        # Level P4 to P5
        bu5 = td5 + self.downsample(bu4)
        # P5 refinement
        bu5 = self.bu_eca3(self.bu_conv3(bu5))
        bu5 = self.bu_c3_3(bu5)
        
        return [bu3, bu4, bu5]

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(C3, self).__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BiFPN_Block(in_channels, hidden_channels)
        self.conv2 = BiFPN_Block(in_channels, hidden_channels)
        self.conv3 = BiFPN_Block(2 * hidden_channels, out_channels)
        
        self.blocks = nn.Sequential(
            *[BiFPN_Block(hidden_channels, hidden_channels) for _ in range(n)]
        )
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.blocks(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class LSF_Detect(nn.Module):
    """Learnable Scalar Fusion Detection Heads"""
    def __init__(self, in_channels, num_classes):
        super(LSF_Detect, self).__init__()
        # LSF modules for each level
        self.lsf3 = LearnableScalarFusion(num_levels=3)
        self.lsf4 = LearnableScalarFusion(num_levels=3)
        self.lsf5 = LearnableScalarFusion(num_levels=3)
        
        # Detection heads
        self.head3 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        self.head4 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        self.head5 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        
    def forward(self, inputs):
        # inputs: [p3, p4, p5]
        p3, p4, p5 = inputs
        
        # Apply Learnable Scalar Fusion
        lsf3 = self.lsf3([p3, p4, p5])
        lsf4 = self.lsf4([p3, p4, p5])
        lsf5 = self.lsf5([p3, p4, p5])
        
        # Detection outputs
        out3 = self.head3(lsf3)
        out4 = self.head4(lsf4)
        out5 = self.head5(lsf5)
        
        return [out3, out4, out5]

# Updated class names for new dataset
CLASS_NAMES = ['hot_spot', 'scratch', 'no_electricity', 'black_border', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class SolarClient:
    def __init__(self, client_id, base_path, server_ip='localhost', port=12000):
        self.client_id = client_id
        self.server_ip = server_ip
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Client {client_id} using device: {self.device}")
        print(f"Client {client_id} initializing...")
        
        # Setup paths
        self.base_path = base_path
        self.images_path = os.path.join(base_path, "images")
        self.annotations_path = os.path.join(base_path, "labels")
        
        # Dataset split paths
        self.train_images = os.path.join(self.base_path, 'train', 'images')
        self.train_labels = os.path.join(self.base_path, 'train', 'labels')
        self.val_images = os.path.join(self.base_path, 'val', 'images')
        self.val_labels = os.path.join(self.base_path, 'val', 'labels')
        self.split_info_file = os.path.join(self.base_path, f'split_info_client{client_id}.json')
        self.dataset_prepared = False

        # Load base YOLO model
        print(f"Loading YOLO model for client {client_id}...")
        self.model = YOLO("yolo11n.pt")
        
        # Count original parameters
        total_params, trainable_params = count_parameters(self.model.model)
        print(f"Original model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        self.modify_model()
        
        # Count modified parameters
        total_params, trainable_params = count_parameters(self.model.model)
        print(f"Modified model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print(f"Client {client_id} model initialized with MobileNetV3Block, ECA, and Learnable Scalar Fusion!")

    def inspect_yolo_layers(self):
        """Inspect YOLOv11 model structure"""
        model = self.model.model
        print("\n" + "="*80)
        print("YOLOv11 MODEL STRUCTURE INSPECTION")
        print("="*80)
        
        # Print backbone layers
        print("\nBackbone Layers (0-8):")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i <= 8:
                print(f"{i}: {name} - {module.__class__.__name__}")
                if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    print(f"    in_channels: {module.in_channels}, out_channels: {module.out_channels}")
        
        # Print neck layers (typically after backbone)
        print("\nNeck Layers (after 8):")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i > 8:
                print(f"{i}: {name} - {module.__class__.__name__}")
                if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    print(f"    in_channels: {module.in_channels}, out_channels: {module.out_channels}")
        
        # Check for Detect head
        print("\nLooking for Detect head...")
        for name, module in model.named_children():
            if 'detect' in name.lower():
                print(f"Found Detect head: {name} - {module.__class__.__name__}")
        
        print("="*80 + "\n")

    def modify_model(self):
        """Modify YOLO model with MobileNetV3Block, ECA, and LSF - REMOVE original neck/head"""
        model = self.model.model
        
        # First inspect the model structure
        self.inspect_yolo_layers()
        
        model.nc = NUM_CLASSES
        model.names = CLASS_NAMES
        
        # ===========================================
        # STEP 1: Replace layer at index 8 with MobileNetV3Block
        # ===========================================
        if len(list(model.children())) > 8:
            # Create new sequential with MobileNetV3Block at position 8
            layers = list(model.children())
            
            # Replace layer 8 with MobileNetV3Block
            layers[8] = MobileNetV3Block(256, 256).to(self.device)
            
            # Create new model with modified layers
            model.model = nn.Sequential(*layers)
        
        # ===========================================
        # STEP 2: Remove original neck layers (typically indices 9-20 in YOLO)
        # ===========================================
        print("Removing original neck layers...")
        layers = list(model.model.children())
        
        # Remove neck layers (indices 9 to around 20, depending on YOLOv11 structure)
        # YOLO neck typically includes SPP, PAN, Upsample, Concat layers
        neck_start_idx = 9
        neck_end_idx = min(25, len(layers))  # Conservative estimate
        
        for i in range(neck_start_idx, neck_end_idx):
            if i < len(layers):
                layers[i] = nn.Identity()
                print(f"  Replaced layer {i} with Identity")
        
        model.model = nn.Sequential(*layers)
        
        # ===========================================
        # STEP 3: Remove original Detect head if exists
        # ===========================================
        print("Removing original Detect head...")
        for name, module in model.named_children():
            if 'detect' in name.lower():
                print(f"  Found and removing {name}")
                setattr(model, name, nn.Identity())
        
        # ===========================================
        # STEP 4: Add our custom modules
        # ===========================================
        print("Adding custom modules...")
        
        # Feature reduction layers with GhostConv
        model.lat_conv3 = LateralGhostConv(64, 128).to(self.device)
        model.lat_conv4 = LateralGhostConv(128, 128).to(self.device)
        model.lat_conv5 = LateralGhostConv(256, 128).to(self.device)
        
        # BiFPN with ECA attention
        model.bifpn_eca = BiFPN_ECA(128).to(self.device)
        
        # Learnable Scalar Fusion Detection Heads
        model.lsf_detect = LSF_Detect(128, NUM_CLASSES).to(self.device)
        
        # ===========================================
        # STEP 5: Replace forward method
        # ===========================================
        def custom_forward(x):
            # Get backbone features
            y = []
            for i, m in enumerate(model.model):
                if hasattr(m, 'f') and m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)  # run
                y.append(x if hasattr(m, 'i') and m.i in getattr(model, 'save', []) else None)
            
            # Extract feature maps at specific indices
            # p3=4, p4=6, p5=8 (now MobileNetV3Block at index 8)
            p3 = y[4] if len(y) > 4 else None
            p4 = y[6] if len(y) > 6 else None
            p5 = y[8] if len(y) > 8 else None
            
            if p3 is None or p4 is None or p5 is None:
                print(f"Warning: Could not extract feature maps. p3={p3}, p4={p4}, p5={p5}")
                return None
            
            # Apply lateral convolutions (GhostConv)
            p3 = model.lat_conv3(p3)
            p4 = model.lat_conv4(p4)
            p5 = model.lat_conv5(p5)
            
            # Pass through BiFPN-ECA
            features = model.bifpn_eca([p3, p4, p5])
            
            # Pass through LSF detection heads
            return model.lsf_detect(features)
        
        model.forward = custom_forward
        model.to(self.device)
        
        # ===========================================
        # STEP 6: Clean up unused parameters
        # ===========================================
        print("Cleaning up unused parameters...")
        
        # Remove Identity layers from parameter list
        for name, module in model.named_children():
            if isinstance(module, nn.Identity):
                # Identity layers have no parameters, but we should ensure they're not in state_dict
                for param_name, param in list(module.named_parameters(recurse=False)):
                    param_name_full = f"{name}.{param_name}" if param_name else name
                    if param_name_full in model.state_dict():
                        del model.state_dict()[param_name_full]
        
        # Verify no Identity layers in forward pass
        print("\nModel structure after modification:")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i <= 15:  # Show first 16 layers
                print(f"{i}: {name} - {module.__class__.__name__}")
        
        print("\nCustom modules added:")
        print(f"  - lat_conv3: {model.lat_conv3.__class__.__name__}")
        print(f"  - lat_conv4: {model.lat_conv4.__class__.__name__}")
        print(f"  - lat_conv5: {model.lat_conv5.__class__.__name__}")
        print(f"  - bifpn_eca: {model.bifpn_eca.__class__.__name__}")
        print(f"  - lsf_detect: {model.lsf_detect.__class__.__name__}")

    def prepare_dataset(self):
        """Create dataset split only once and save the split information"""
        if self.dataset_prepared:
            print(f"Client {self.client_id} dataset already prepared")
            return True
            
        print(f"Client {self.client_id} preparing dataset (one-time operation)...")
        
        # Create directories
        os.makedirs(self.train_images, exist_ok=True)
        os.makedirs(self.train_labels, exist_ok=True)
        os.makedirs(self.val_images, exist_ok=True)
        os.makedirs(self.val_labels, exist_ok=True)
        
        # Clear existing files only on first run
        for folder in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        # Collect all valid image-label pairs
        valid_images = []
        for label_file in os.listdir(self.annotations_path):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                img_found = False
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_path = os.path.join(self.images_path, base_name + ext)
                    if os.path.exists(img_path):
                        valid_images.append((base_name, ext))
                        img_found = True
                        break
        
        # Create validation split (10%) only once
        val_files = set()
        if os.path.exists(self.split_info_file):
            # Load existing split information
            with open(self.split_info_file, 'r') as f:
                val_files = set(json.load(f))
            print(f"Loaded existing split info for client {self.client_id}")
        else:
            # Create new split
            if len(valid_images) > 5:
                n_val = max(1, int(0.1 * len(valid_images)))
                val_indices = random.sample(range(len(valid_images)), n_val)
                val_files = {valid_images[i][0] for i in val_indices}
            
            # Save split information for future runs
            with open(self.split_info_file, 'w') as f:
                json.dump(list(val_files), f)
            print(f"Created new split for client {self.client_id}")
        
        # Copy files to appropriate directories using the saved split
        for base_name, ext in valid_images:
            img_src = os.path.join(self.images_path, base_name + ext)
            label_src = os.path.join(self.annotations_path, base_name + '.txt')
            
            if base_name in val_files:
                # Copy to validation
                shutil.copy2(img_src, os.path.join(self.val_images, base_name + ext))
                shutil.copy2(label_src, os.path.join(self.val_labels, base_name + '.txt'))
            else:
                # Copy to training
                shutil.copy2(img_src, os.path.join(self.train_images, base_name + ext))
                shutil.copy2(label_src, os.path.join(self.train_labels, base_name + '.txt'))
        
        print(f"Client {self.client_id} dataset prepared")
        self.dataset_prepared = True
        return True

    def create_yaml(self):
        """Create dataset YAML once"""
        yaml_content = {
            'path': os.path.abspath(self.base_path),
            'train': 'train',
            'val': 'val',
            'names': CLASS_NAMES,
            'nc': NUM_CLASSES
        }
        yaml_path = os.path.join(self.base_path, f'client_{self.client_id}.yaml')
        
        if not os.path.exists(yaml_path):
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f)
        
        return yaml_path

    def local_train(self, epochs=10, project=None, name='train'):
        try:
            # Prepare dataset (only once)
            if not self.dataset_prepared:
                self.prepare_dataset()
            
            # Create YAML (only once)
            yaml_path = self.create_yaml()
            
            # Determine device for training
            device_str = '0' if self.device.type == 'cuda' else 'cpu'
            
            # Train the model with GPU acceleration
            self.model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=device_str,
                verbose=True,
                project=project,
                name=name,
                exist_ok=True,
                augment=True,
                patience=5,
                half=False,
                workers=0   
            )
            
            print(f"Training completed for {epochs} epochs")
            return True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            traceback.print_exc()
            return False

    def get_non_bn_weights(self):
        """Extract only non-BN weights for federated sharing"""
        state_dict = self.model.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items() if not any(
            bn_term in k for bn_term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
        )}

    def update_with_global_weights(self, global_weights):
        """Update model with global weights while preserving local BN layers"""
        # Get current state
        client_state = self.model.model.state_dict()
        
        # Update only non-BN layers
        for key in client_state:
            if not any(bn_term in key for bn_term in 
                      ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']):
                if key in global_weights and client_state[key].shape == global_weights[key].shape:
                    client_state[key] = global_weights[key].to(self.device)
        
        # Load updated state
        self.model.model.load_state_dict(client_state, strict=False)

    def connect_to_server(self, action='send_weights', round_idx=1):
        """Generic connection method for both phases"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(30000)  # 5-minute timeout
                sock.connect((self.server_ip, self.port))
                
                # Send client ID first
                sock.sendall(f"CLIENT{self.client_id}".encode())
                
                if action == 'send_weights':
                    # Receive server signal to start
                    signal = sock.recv(5).decode()  # Only need 5 bytes for "START"
                    if signal != "START":
                        print(f"Unexpected server signal: {signal}")
                        return False
                    
                    # Local training
                    print(f"\n--- Round {round_idx} Local Training ---")
                    project = f"runs/client{self.client_id}"
                    if not self.local_train(
                        epochs=10,
                        project=project,
                        name=f"round{round_idx}"
                    ):
                        return False
                    
                    # Send weights to server (only non-BN weights)
                    non_bn_weights = self.get_non_bn_weights()
                    data = pickle.dumps(non_bn_weights)
                    
                    # Send length first
                    sock.sendall(len(data).to_bytes(4, 'big'))
                    # Send data in chunks
                    total_sent = 0
                    while total_sent < len(data):
                        chunk = data[total_sent:total_sent+4096]
                        sent = sock.send(chunk)
                        if sent == 0:
                            raise RuntimeError("Socket connection broken")
                        total_sent += sent
                    print(f"Sent weights to server for round {round_idx}")
                    return True
                
                elif action == 'receive_model':
                    # Receive global model
                    header = sock.recv(4)
                    if not header:
                        print("No header received from server")
                        return False
                    msglen = int.from_bytes(header, 'big')
                    
                    received = b''
                    while len(received) < msglen:
                        remaining = msglen - len(received)
                        packet = sock.recv(min(4096, remaining))
                        if not packet:
                            break
                        received += packet
                    
                    if len(received) != msglen:
                        print(f"Incomplete global model: {len(received)}/{msglen} bytes")
                        return False
                    
                    # Load using torch.load from bytes buffer
                    try:
                        buffer = io.BytesIO(received)
                        global_weights = torch.load(buffer, map_location='cpu')
                        print(f"Received global weights for round {round_idx}")
                        self.update_with_global_weights(global_weights)
                        return True
                    except Exception as e:
                        print(f"Error loading global weights: {str(e)}")
                        traceback.print_exc()
                        return False
                
            except socket.timeout:
                print("Connection timed out")
                return False
            except ConnectionResetError:
                print("Connection reset by server")
                return False
            except Exception as e:
                print(f"Connection error: {str(e)}")
                traceback.print_exc()
                return False
        return False

    def run(self, rounds=5):
        # Prepare dataset once at the beginning
        self.prepare_dataset()
        self.create_yaml()
        
        for round_idx in range(1, rounds+1):
            print(f"\n=== Client {self.client_id} - Round {round_idx}/{rounds} ===")
            
            # Phase 1: Train and send weights to server
            print("Connecting to server to send weights...")
            if not self.connect_to_server(action='send_weights', round_idx=round_idx):
                print(f"Failed to send weights in round {round_idx}")
                return
                
            # Phase 2: Receive updated global model from server
            print("Connecting to server to receive updated model...")
            if not self.connect_to_server(action='receive_model', round_idx=round_idx):
                print(f"Failed to receive global model in round {round_idx}")
                return
                
        print("Federated learning completed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <client_id>")
        sys.exit(1)
    
    client_id = int(sys.argv[1])
    
    if client_id == 1:
        base_path = r"E:\D14 PV Multi Defect_split\D14 PV Multi Defect_split\FederatedSplit1\yolo_client1"
    elif client_id == 2:
        base_path = r"E:\D14 PV Multi Defect_split\D14 PV Multi Defect_split\FederatedSplit1\yolo_client2"
    else:
        print("Invalid client ID. Use 1 or 2.")
        sys.exit(1)

    client = SolarClient(
        client_id=client_id,  
        base_path=base_path,
        server_ip='localhost',
        port=12000
    )
    client.run(rounds=7)