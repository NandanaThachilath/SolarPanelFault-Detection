import socket
import pickle
import torch
import os
import yaml
import shutil
import time
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import threading
import traceback
import numpy as np

# Custom Modules (same as client)
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

class FederatedServer:
    def __init__(self, num_clients=2, port=12000):
        self.num_clients = num_clients
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("Initializing global model...")
        
        # Load base YOLO model
        self.global_model = YOLO("yolo11n.pt")
        
        # Count original parameters
        total_params, trainable_params = count_parameters(self.global_model.model)
        print(f"Original model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        self.modify_model()
        
        # Count modified parameters
        total_params, trainable_params = count_parameters(self.global_model.model)
        print(f"Modified model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print("Global model initialized with MobileNetV3Block, ECA, and Learnable Scalar Fusion!")
        
        # Create dummy dataset directory
        self.dummy_dir = "server_dummy_data"
        os.makedirs(self.dummy_dir, exist_ok=True)
        self.create_dummy_dataset()

    def inspect_yolo_layers(self):
        """Inspect YOLOv11 model structure"""
        model = self.global_model.model
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
        model = self.global_model.model
        
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

    def create_dummy_dataset(self):
        """Create minimal dataset structure for YOLO initialization"""
        # Create directory structure
        train_img_dir = os.path.join(self.dummy_dir, 'train', 'images')
        train_lbl_dir = os.path.join(self.dummy_dir, 'train', 'labels')
        val_img_dir = os.path.join(self.dummy_dir, 'val', 'images')
        val_lbl_dir = os.path.join(self.dummy_dir, 'val', 'labels')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)

        # Create dummy image
        dummy_img = Image.new('RGB', (64, 64), color='black')
        dummy_img_path = os.path.join(train_img_dir, 'dummy.jpg')
        dummy_img.save(dummy_img_path)
        
        # Create dummy label
        with open(os.path.join(train_lbl_dir, 'dummy.txt'), 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        
        # Copy to validation
        shutil.copy(dummy_img_path, os.path.join(val_img_dir, 'dummy.jpg'))
        shutil.copy(os.path.join(train_lbl_dir, 'dummy.txt'), 
                    os.path.join(val_lbl_dir, 'dummy.txt'))

        # Create dataset config
        dummy_data = {
            'path': os.path.abspath(self.dummy_dir),
            'train': 'train',
            'val': 'val',
            'nc': NUM_CLASSES,
            'names': CLASS_NAMES
        }
        self.dummy_yaml = os.path.join(self.dummy_dir, 'dummy.yaml')
        with open(self.dummy_yaml, 'w') as f:
            yaml.dump(dummy_data, f)

    def reset_round(self):
        self.client_weights = {}
        self.remaining_clients = set(range(1, self.num_clients+1))

    def federated_averaging(self):
        print("\nPerforming federated averaging...")
        global_dict = self.global_model.model.state_dict()
        
        # Collect all layer keys
        all_keys = set(global_dict.keys())
        for weights in self.client_weights.values():
            all_keys.update(weights.keys())
            
        avg_weights = {}
        for key in all_keys:
            # Skip BN layers for FedBN
            if any(bn_term in key for bn_term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']):
                continue
                
            weight_list = []
            for client_id in range(1, self.num_clients+1):
                if client_id in self.client_weights and key in self.client_weights[client_id]:
                    weight_list.append(self.client_weights[client_id][key].to(self.device))
            
            if weight_list:
                if len(weight_list) == self.num_clients:
                    avg_weights[key] = torch.stack(weight_list, dim=0).mean(0)
                else:
                    print(f"Key {key} missing in some clients, using global weights")
                    avg_weights[key] = global_dict[key].to(self.device)
        
        # Update global model
        new_state_dict = global_dict.copy()
        for k, v in avg_weights.items():
            if k in new_state_dict and v.shape == new_state_dict[k].shape:
                new_state_dict[k] = v
                
        self.global_model.model.load_state_dict(new_state_dict, strict=False)
        print("Federated averaging completed!")
        return True

    def handle_client(self, conn, addr, round_idx):
        try:
            print(f"Connected to {addr}")
            client_id_str = conn.recv(7).decode()
            if not client_id_str.startswith("CLIENT"):
                print("Invalid client ID format")
                return None
            
            client_id = int(client_id_str[6:])
            print(f"Received connection from client {client_id}")
            
            # Send start signal
            conn.sendall("START".encode())
            
            # Receive client weights
            header = conn.recv(4)
            if not header:
                print("No header received")
                return None
            msglen = int.from_bytes(header, 'big')
            
            received = b''
            while len(received) < msglen:
                packet = conn.recv(min(4096, msglen - len(received)))
                if not packet:
                    break
                received += packet
            
            if len(received) != msglen:
                print(f"Incomplete data: {len(received)}/{msglen} bytes")
                return None
            
            # Deserialize with error handling
            try:
                client_weights = pickle.loads(received)
                print(f"Received weights from client {client_id}")
                return client_id, client_weights
            except Exception as e:
                print(f"Error deserializing weights: {str(e)}")
                return None
                
        except Exception as e:
            print(f"Error handling client: {str(e)}")
            traceback.print_exc()
            return None
        finally:
            try:
                conn.close()
            except:
                pass

    def send_global_model(self, conn):
        """Send global model to client"""
        try:
            # Serialize with error handling
            buffer = io.BytesIO()
            torch.save(self.global_model.model.state_dict(), buffer)
            global_model_bytes = buffer.getvalue()
            
            # Send length first
            conn.sendall(len(global_model_bytes).to_bytes(4, 'big'))
            # Send data in chunks
            total_sent = 0
            while total_sent < len(global_model_bytes):
                chunk = global_model_bytes[total_sent:total_sent+4096]
                sent = conn.send(chunk)
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent
            return True
        except Exception as e:
            print(f"Error sending global model: {str(e)}")
            traceback.print_exc()
            return False

    def start(self, rounds=5):
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(10)
            print(f"Server listening on port {self.port}...")

            for round_idx in range(1, rounds+1):
                print(f"\n===== Starting Round {round_idx}/{rounds} =====")
                self.reset_round()
                
                # Phase 1: Collect weights from all clients
                print(f"Waiting for {self.num_clients} clients to submit weights...")
                for client_num in range(self.num_clients):
                    conn, addr = server_socket.accept()
                    print(f"Accepted connection from {addr}")
                    result = self.handle_client(conn, addr, round_idx)
                    if result:
                        client_id, weights = result
                        self.client_weights[client_id] = weights
                        self.remaining_clients.discard(client_id)
                        print(f"Collected weights from client {client_id}")
                
                # Perform aggregation if we have all clients
                if len(self.client_weights) > 0:
                    self.federated_averaging()
                else:
                    print("No client weights received, skipping aggregation")
                
                # Save checkpoint
                save_path = f'global_round_{round_idx}.pt'
                torch.save(self.global_model.model.state_dict(), save_path)
                print(f"Saved global weights to {save_path}")
                
                # Phase 2: Distribute updated model to clients
                print("\nDistributing updated global model to clients...")
                clients_to_send = set(range(1, self.num_clients+1))
                
                while clients_to_send:
                    print(f"Waiting for {len(clients_to_send)} clients to receive model...")
                    conn, addr = server_socket.accept()
                    print(f"Accepted connection from {addr} for model distribution")
                    
                    try:
                        client_id_str = conn.recv(7).decode()
                        if client_id_str.startswith("CLIENT"):
                            client_id = int(client_id_str[6:])
                            if client_id in clients_to_send:
                                print(f"Sending global model to client {client_id}")
                                if self.send_global_model(conn):
                                    clients_to_send.remove(client_id)
                                    print(f"Successfully sent model to client {client_id}")
                                else:
                                    print(f"Failed to send model to client {client_id}")
                            else:
                                print(f"Unexpected client ID: {client_id}")
                        else:
                            print(f"Invalid client ID format: {client_id_str}")
                    except Exception as e:
                        print(f"Error during model distribution: {str(e)}")
                    finally:
                        try:
                            conn.close()
                        except:
                            pass
                
                print(f"===== Round {round_idx} Completed =====")
                
        except Exception as e:
            print(f"Server error: {str(e)}")
            traceback.print_exc()
        finally:
            # Clean up dummy data at the very end
            try:
                shutil.rmtree(self.dummy_dir, ignore_errors=True)
                print("Cleaned up dummy data")
            except Exception as e:
                print(f"Error cleaning dummy data: {str(e)}")
            try:
                server_socket.close()
            except:
                pass
            print("Server shutdown")

if __name__ == "__main__":
    server = FederatedServer(num_clients=2, port=12000)
    server.start(rounds=7)