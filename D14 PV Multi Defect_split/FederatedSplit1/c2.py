import socket
import pickle
import torch
import os
import shutil
import random
from ultralytics import YOLO
import yaml
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import time
import json
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import traceback

# ==================== Custom Modules ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ASFF(nn.Module):
    def __init__(self, level, multiplier=4):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = 128
        self.inter_dim = self.dim // multiplier
        
        # Level-specific weights
        self.weight_levels = nn.Conv2d(self.dim * 3, 3, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x: list of feature maps [p3, p4, p5]
        target_size = x[self.level].size()[2:]
        
        # Resize all features to target level
        resized = []
        for i, feat in enumerate(x):
            if i < self.level:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            elif i > self.level:
                feat = F.avg_pool2d(feat, kernel_size=2**(i-self.level), stride=2**(i-self.level))
            resized.append(feat)
        
        # Concatenate features
        fused = torch.cat(resized, dim=1)
        
        # Compute weights
        weights = self.weight_levels(fused)
        weights = self.softmax(weights)
        
        # Weighted sum
        out = 0
        for i in range(3):
            out += weights[:, i:i+1] * resized[i]
            
        return out

class BiFPN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.conv(x))

class BiFPN_CBAM(nn.Module):
    def __init__(self, channels=128):
        super(BiFPN_CBAM, self).__init__()
        # Feature scaling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Top-down pathway
        self.td_conv1 = BiFPN_Block(channels, channels)
        self.td_cbam1 = CBAM(channels)
        self.td_c3_1 = C3(channels, channels, n=2)
        
        self.td_conv2 = BiFPN_Block(channels, channels)
        self.td_cbam2 = CBAM(channels)
        self.td_c3_2 = C3(channels, channels, n=2)
        
        # P3 refinement
        self.td_conv3 = BiFPN_Block(channels, channels)
        self.td_cbam3 = CBAM(channels)
        self.td_c3_3 = C3(channels, channels, n=2)
        
        # Bottom-up pathway
        self.bu_conv1 = BiFPN_Block(channels, channels)
        self.bu_cbam1 = CBAM(channels)
        self.bu_c3_1 = C3(channels, channels, n=2)
        
        self.bu_conv2 = BiFPN_Block(channels, channels)
        self.bu_cbam2 = CBAM(channels)
        self.bu_c3_2 = C3(channels, channels, n=2)
        
        # P5 refinement
        self.bu_conv3 = BiFPN_Block(channels, channels)
        self.bu_cbam3 = CBAM(channels)
        self.bu_c3_3 = C3(channels, channels, n=2)
        
    def forward(self, inputs):
        # Unpack inputs (P3, P4, P5)
        p3, p4, p5 = inputs
        
        # ========== Top-down pathway ==========
        # Level P5 (level 5)
        td5 = p5
        td5 = self.td_cbam1(self.td_conv1(td5))
        td5 = self.td_c3_1(td5)
        
        # Level P5 to P4
        td4 = p4 + self.upsample(td5)
        td4 = self.td_cbam2(self.td_conv2(td4))
        td4 = self.td_c3_2(td4)
        
        # Level P4 to P3
        td3 = p3 + self.upsample(td4)
        # P3 refinement
        td3 = self.td_cbam3(self.td_conv3(td3))
        td3 = self.td_c3_3(td3)
        
        # ========== Bottom-up pathway ==========
        # Level P3 (level 3)
        bu3 = td3
        bu3 = self.bu_cbam1(self.bu_conv1(bu3))
        bu3 = self.bu_c3_1(bu3)
        
        # Level P3 to P4
        bu4 = td4 + self.downsample(bu3)
        bu4 = self.bu_cbam2(self.bu_conv2(bu4))
        bu4 = self.bu_c3_2(bu4)
        
        # Level P4 to P5
        bu5 = td5 + self.downsample(bu4)
        # P5 refinement
        bu5 = self.bu_cbam3(self.bu_conv3(bu5))
        bu5 = self.bu_c3_3(bu5)
        
        return [bu3, bu4, bu5]

class LateralConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LateralConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        return self.conv(x)

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

class ASFF_Detect(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ASFF_Detect, self).__init__()
        # ASFF modules for each level
        self.asff3 = ASFF(level=0)
        self.asff4 = ASFF(level=1)
        self.asff5 = ASFF(level=2)
        
        # Detection heads
        self.head3 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        self.head4 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        self.head5 = nn.Conv2d(in_channels, num_classes + 4, kernel_size=1)
        
    def forward(self, inputs):
        # inputs: [p3, p4, p5]
        p3, p4, p5 = inputs
        
        # Apply ASFF
        asff3 = self.asff3([p3, p4, p5])
        asff4 = self.asff4([p3, p4, p5])
        asff5 = self.asff5([p3, p4, p5])
        
        # Detection outputs
        out3 = self.head3(asff3)
        out4 = self.head4(asff4)
        out5 = self.head5(asff5)
        
        return [out3, out4, out5]

# ==================== Federated Client ====================
CLASS_NAMES = [
    'hot_spot', 'scratch', 'black_border', 
    'no_electricity', 'broken'
]
NUM_CLASSES = len(CLASS_NAMES)

class SolarDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=640):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        base_name = os.path.splitext(self.image_files[idx])[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + '.txt')
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img) / 255.0
        
        # Load labels
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.split())
                    labels.append([class_id, x, y, w, h])
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5))

class SolarClient:
    def __init__(self, client_id, annotations_path, images_path, server_ip='localhost', port=12000):
        self.client_id = client_id
        self.server_ip = server_ip
        self.port = port
        
        # Check for GPU availability
        self.device = torch.device('cpu')
        torch.set_num_threads(4)
        print(f"Client {client_id} using device: {self.device}")
        print(f"Client {client_id} initializing...")
        
        # Load base YOLO model
        self.model = YOLO("yolo11n.pt")
        self.modify_model()
        print(f"Client {client_id} model initialized with BiFPN-CBAM and ASFF!")
        
        # Setup paths
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.base_path = os.path.dirname(images_path)
        self.round_delay = 1 if client_id == 1 else 2  # Stagger client starts
        
        # Paths for dataset
        self.train_images = os.path.join(self.base_path, 'train', 'images')
        self.train_labels = os.path.join(self.base_path, 'train', 'labels')
        self.val_images = os.path.join(self.base_path, 'val', 'images')
        self.val_labels = os.path.join(self.base_path, 'val', 'labels')
        self.split_info_file = os.path.join(self.base_path, f'split_info_client{client_id}.json')
        
        # Flag to track if dataset is prepared
        self.dataset_prepared = False

    def modify_model(self):
        """Modify YOLO model with custom neck and head"""
        model = self.model.model
        model.nc = NUM_CLASSES
        model.names = CLASS_NAMES
        
        # Feature reduction layers
        model.lat_conv3 = LateralConv(64, 128).to(self.device)
        model.lat_conv4 = LateralConv(128, 128).to(self.device)
        model.lat_conv5 = LateralConv(256, 128).to(self.device)
        
        # BiFPN with CBAM and integrated C3 blocks
        model.bifpn_cbam = BiFPN_CBAM(128).to(self.device)
        
        # ASFF Detection Heads
        model.asff_detect = ASFF_Detect(128, NUM_CLASSES).to(self.device)
        
        # Replace forward method
        def custom_forward(x):
            y = []
            for i, m in enumerate(model.model):
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in model.save else None)  # save output
            p3 = y[3]  # Output from layer 4
            p4 = y[5]  # Output from layer 6
            p5 = y[8]  # Output from layer 9
            
            # Apply lateral convolutions
            p3 = model.lat_conv3(p3)
            p4 = model.lat_conv4(p4)
            p5 = model.lat_conv5(p5)
            
            # Pass through BiFPN-CBAM
            features = model.bifpn_cbam([p3, p4, p5])
            
            # Pass through ASFF detection heads
            return model.asff_detect(features)
        
        model.forward = custom_forward
        model.to(self.device)

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
                    os.remove(file_path)
        
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
            
            # Train the model
            self.model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=600,
                optimizer='AdamW',
                lr0=0.000625,           # initial learning rate
                momentum=0.9,           # for AdamW that's the beta1 parameter
                weight_decay=0.0005,
                batch=16,
                device='cpu',
                verbose=True,
                project=project,
                name=name,
                exist_ok=True,
                augment=True,
                
                patience=5,
                half=False,  # Explicitly disable mixed precision
                workers=0    # Reduce workers to avoid multiprocessing issues
            )
            
            print(f"Training completed for {epochs} epochs")
            return True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            print(traceback.format_exc())
            return False

    def get_non_bn_weights(self):
        """Extract only non-BN weights for federated sharing"""
        state_dict = self.model.model.state_dict()
        
        # Return only non-BN weights (move to CPU for transmission)
        return {k: v.cpu() for k, v in state_dict.items() if not any(
            bn_term in k for bn_term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
        )}

    def update_with_global_weights(self, global_path):
        """Update model with global weights while preserving local BN layers"""
        # Load global weights (always on CPU)
        global_state = torch.load(global_path, map_location='cpu')
        
        # Get current state
        client_state = self.model.model.state_dict()
        
        # Update only non-BN layers
        for key in client_state:
            if not any(bn_term in key for bn_term in 
                      ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']):
                if key in global_state and client_state[key].shape == global_state[key].shape:
                    # Move tensor to correct device
                    client_state[key] = global_state[key].to(self.device)
        
        # Load updated state
        self.model.model.load_state_dict(client_state, strict=False)

    def federated_round(self, round_idx):
        # Add delay to stagger client starts
        time.sleep(self.round_delay)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                # Connect to server with timeout
                sock.settimeout(36000)  # 10-hour timeout
                sock.connect((self.server_ip, self.port))
                
                # Send client ID first
                sock.sendall(f"CLIENT{self.client_id}".encode())
                
                # Local training
                print(f"\n--- Round {round_idx} Local Training ---")
                project = f"runs/client{self.client_id}"
                self.local_train(
                    epochs=10,  # Reduced epochs for testing
                    project=project,
                    name=f"round{round_idx}"
                )
                
                # Send weights to server (only non-BN weights)
                non_bn_weights = self.get_non_bn_weights()
                data = pickle.dumps(non_bn_weights)
                sock.sendall(len(data).to_bytes(4, 'big'))
                sock.sendall(data)
                print(f"Sent weights to server for round {round_idx}")
                
                # Receive global model
                header = sock.recv(4)
                if not header:
                    print("No header received from server")
                    return False
                msglen = int.from_bytes(header, 'big')
                
                received = b''
                while len(received) < msglen:
                    try:
                        packet = sock.recv(4096)
                        if not packet:
                            break
                        received += packet
                    except socket.timeout:
                        print("Socket timeout during global model reception")
                        break
                
                if len(received) != msglen:
                    print(f"Incomplete global model: {len(received)}/{msglen} bytes")
                    return False
                
                # Save global weights
                global_path = f'global_client{self.client_id}_round{round_idx}.pt'
                with open(global_path, 'wb') as f:
                    f.write(received)
                
                # Update model with global weights (preserving local BN)
                print(f"Updating model with global weights for round {round_idx}")
                self.update_with_global_weights(global_path)
                return True
                
            except socket.timeout as te:
                print(f"Socket timeout during round {round_idx}: {str(te)}")
                return False
            except ConnectionResetError as cre:
                print(f"Connection reset during round {round_idx}: {str(cre)}")
                return False
            except Exception as e:
                print(f"Error during round {round_idx}: {str(e)}")
                return False
            finally:
                # Add delay to prevent connection flooding
                time.sleep(1)

    def run(self, rounds=5):
        # Prepare dataset once at the beginning
        self.prepare_dataset()
        self.create_yaml()
        
        for round_idx in range(1, rounds+1):
            print(f"\n=== Client {self.client_id} - Round {round_idx}/{rounds} ===")
            success = self.federated_round(round_idx)
            if not success:
                print(f"Round {round_idx} failed, skipping further rounds")
                return
                
        print("Federated learning completed!")

if __name__ == "__main__":
    client_id = 2# Change to 1 for first client
    client = SolarClient(
        client_id=client_id,  
        images_path=r"C:\Users\Admin\Documents\D14 PV Multi Defect_split\FederatedSplit1\yolo_client2\images",
        annotations_path=r"C:\Users\Admin\Documents\D14 PV Multi Defect_split\FederatedSplit1\yolo_client2\labels",
        server_ip='localhost',
        port=12000
    )
    client.run(rounds=5)