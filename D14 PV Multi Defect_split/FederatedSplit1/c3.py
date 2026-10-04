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

# Updated class names for new dataset
CLASS_NAMES = ['hot_spot', 'scratch', 'no_electricity', 'black_border', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

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
        self.model = YOLO("yolo11n.pt")
        self.modify_model()
        print(f"Client {client_id} model initialized with BiFPN-CBAM and ASFF!")

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
            # Updated indices: p3=4, p4=6, p5=8 (C2k2 layers)
            p3 = y[4]  
            p4 = y[6]  
            p5 = y[8]  
            
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

    def generate_enhanced_confusion_matrix(self, cm, save_path):
        """
        Generate enhanced confusion matrix with custom styling
        """
        try:
            plt.figure(figsize=(14, 12))
            ax = sns.heatmap(
                cm, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues', 
                cbar=False,
                annot_kws={'size': 12}
            )
            
            # Set labels with custom styling
            ax.set_xticklabels(
                CLASS_NAMES, 
                fontsize=18, 
                rotation=45, 
                ha='right',
                rotation_mode='anchor'
            )
            ax.set_yticklabels(
                CLASS_NAMES, 
                fontsize=16, 
                rotation=0,
                va='center'
            )
            
            # Add axis labels with increased font size
            plt.xlabel('Predicted Labels', fontsize=20, labelpad=20)
            plt.ylabel('True Labels', fontsize=20, labelpad=20)
            
            # Add title
            plt.title(f'Client {self.client_id} - Confusion Matrix', fontsize=22, pad=20)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved enhanced confusion matrix at: {save_path}")
        except Exception as e:
            print(f"Error generating enhanced confusion matrix: {str(e)}")
            traceback.print_exc()

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
            
            # Generate enhanced confusion matrix
            val_dir = os.path.join(project, name, 'val')
            cm_path = os.path.join(val_dir, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                # Load existing confusion matrix
                cm_img = plt.imread(cm_path)
                cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
                
                # Extract values from image (simplified approach)
                # In practice, you'd need to parse the actual matrix from metrics
                # This is a placeholder for the actual confusion matrix data
                
                # Generate enhanced version
                enhanced_path = os.path.join(val_dir, f'confusion_matrix_enhanced_round{name.split("round")[-1]}.png')
                self.generate_enhanced_confusion_matrix(cm, enhanced_path)
            
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