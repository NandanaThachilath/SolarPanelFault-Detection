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

# ================== Custom Modules ==================
class ECA(nn.Module):
    """Efficient Channel Attention Module"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 use_se=True, activation='hswish'):
        super(MobileNetV3Block, self).__init__()
        self.use_se = use_se
        self.stride = stride
        expanded_channels = in_channels * 6
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        self.expand_act = nn.Hardswish() if activation == 'hswish' else nn.ReLU()
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels,
                                        kernel_size, stride, kernel_size//2,
                                        groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_act = nn.Hardswish() if activation == 'hswish' else nn.ReLU()
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, expanded_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(expanded_channels // 4, expanded_channels, 1),
                nn.Hardsigmoid() if activation == 'hswish' else nn.Sigmoid()
            )
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        identity = x
        out = self.expand_act(self.expand_bn(self.expand_conv(x)))
        out = self.depthwise_act(self.depthwise_bn(self.depthwise_conv(out)))
        if self.use_se:
            out = out * self.se(out)
        out = self.project_bn(self.project_conv(out))
        if self.stride == 1 and identity.shape == out.shape:
            out = out + identity
        return out

class LateralGhostConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LateralGhostConv, self).__init__()
        self.ghost_conv = GhostConv(in_channels, out_channels)
    def forward(self, x):
        return self.ghost_conv(x)

class BiFPN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BiFPN_Block, self).__init__()
        self.conv = GhostConv(in_channels, out_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.conv(x))

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(C3, self).__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BiFPN_Block(in_channels, hidden_channels)
        self.conv2 = BiFPN_Block(in_channels, hidden_channels)
        self.conv3 = BiFPN_Block(2 * hidden_channels, out_channels)
        self.blocks = nn.Sequential(*[BiFPN_Block(hidden_channels, hidden_channels) for _ in range(n)])
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.blocks(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)
        return self.conv3(x)

class BiFPN_ECA(nn.Module):
    def __init__(self, channels=128):
        super(BiFPN_ECA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.td_conv1 = BiFPN_Block(channels, channels)
        self.td_eca1 = ECA(channels)
        self.td_c3_1 = C3(channels, channels, n=2)
        self.td_conv2 = BiFPN_Block(channels, channels)
        self.td_eca2 = ECA(channels)
        self.td_c3_2 = C3(channels, channels, n=2)
        self.td_conv3 = BiFPN_Block(channels, channels)
        self.td_eca3 = ECA(channels)
        self.td_c3_3 = C3(channels, channels, n=2)
        self.bu_conv1 = BiFPN_Block(channels, channels)
        self.bu_eca1 = ECA(channels)
        self.bu_c3_1 = C3(channels, channels, n=2)
        self.bu_conv2 = BiFPN_Block(channels, channels)
        self.bu_eca2 = ECA(channels)
        self.bu_c3_2 = C3(channels, channels, n=2)
        self.bu_conv3 = BiFPN_Block(channels, channels)
        self.bu_eca3 = ECA(channels)
        self.bu_c3_3 = C3(channels, channels, n=2)
    def forward(self, inputs):
        p3, p4, p5 = inputs
        td5 = self.td_eca1(self.td_conv1(p5))
        td5 = self.td_c3_1(td5)
        td4 = p4 + self.upsample(td5)
        td4 = self.td_eca2(self.td_conv2(td4))
        td4 = self.td_c3_2(td4)
        td3 = p3 + self.upsample(td4)
        td3 = self.td_eca3(self.td_conv3(td3))
        td3 = self.td_c3_3(td3)
        bu3 = self.bu_eca1(self.bu_conv1(td3))
        bu3 = self.bu_c3_1(bu3)
        bu4 = td4 + self.downsample(bu3)
        bu4 = self.bu_eca2(self.bu_conv2(bu4))
        bu4 = self.bu_c3_2(bu4)
        bu5 = td5 + self.downsample(bu4)
        bu5 = self.bu_eca3(self.bu_conv3(bu5))
        bu5 = self.bu_c3_3(bu5)
        return [bu3, bu4, bu5]

# ========== NEW: Ghost‑based Decoupled Head ==========
class GhostDecoupledDetect(nn.Module):
    """Ghost-based decoupled detection head for one feature level."""
    def __init__(self, in_channels, num_classes, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2   # reduce channels by half
        self.shared = GhostConv(in_channels, hidden_channels, kernel_size=1)
        self.cls = GhostConv(hidden_channels, num_classes, kernel_size=1)
        self.reg = GhostConv(hidden_channels, 4, kernel_size=1)

    def forward(self, x):
        x = self.shared(x)
        cls = self.cls(x)
        reg = self.reg(x)
        return torch.cat([cls, reg], dim=1)   # output: (num_classes+4) channels

class GhostDecoupledDetectMulti(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_channels=None):
        super().__init__()
        self.head3 = GhostDecoupledDetect(in_channels, num_classes, hidden_channels)
        self.head4 = GhostDecoupledDetect(in_channels, num_classes, hidden_channels)
        self.head5 = GhostDecoupledDetect(in_channels, num_classes, hidden_channels)

    def forward(self, inputs):
        p3, p4, p5 = inputs
        out3 = self.head3(p3)
        out4 = self.head4(p4)
        out5 = self.head5(p5)
        return [out3, out4, out5]

# ================== Client ==================
CLASS_NAMES = ['hot_spot', 'scratch', 'no_electricity', 'black_border', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

def count_parameters(model):
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
        self.base_path = base_path
        self.images_path = os.path.join(base_path, "images")
        self.annotations_path = os.path.join(base_path, "labels")
        self.train_images = os.path.join(self.base_path, 'train', 'images')
        self.train_labels = os.path.join(self.base_path, 'train', 'labels')
        self.val_images = os.path.join(self.base_path, 'val', 'images')
        self.val_labels = os.path.join(self.base_path, 'val', 'labels')
        self.split_info_file = os.path.join(self.base_path, f'split_info_client{client_id}.json')
        self.dataset_prepared = False

        print(f"Loading YOLO model for client {client_id}...")
        self.model = YOLO("yolo11n.pt")
        total_params, trainable_params = count_parameters(self.model.model)
        print(f"Original model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        self.modify_model()
        total_params, trainable_params = count_parameters(self.model.model)
        print(f"Modified model: Total params: {total_params:,}, Trainable: {trainable_params:,}")

    def modify_model(self):
        model = self.model.model
        model.nc = NUM_CLASSES
        model.names = CLASS_NAMES

        # Replace layer 8
        if len(list(model.children())) > 8:
            layers = list(model.children())
            layers[8] = MobileNetV3Block(256, 256).to(self.device)
            model.model = nn.Sequential(*layers)

        # Remove neck
        print("Removing original neck layers...")
        layers = list(model.model.children())
        neck_start_idx = 9
        neck_end_idx = min(25, len(layers))
        for i in range(neck_start_idx, neck_end_idx):
            if i < len(layers):
                layers[i] = nn.Identity()
        model.model = nn.Sequential(*layers)

        # Remove Detect head
        print("Removing original Detect head...")
        for name, module in model.named_children():
            if 'detect' in name.lower():
                setattr(model, name, nn.Identity())

        # Add custom modules
        print("Adding custom modules...")
        model.lat_conv3 = LateralGhostConv(64, 128).to(self.device)
        model.lat_conv4 = LateralGhostConv(128, 128).to(self.device)
        model.lat_conv5 = LateralGhostConv(256, 128).to(self.device)
        model.bifpn_eca = BiFPN_ECA(128).to(self.device)
        # Ghost‑based decoupled head
        model.lsf_detect = GhostDecoupledDetectMulti(128, NUM_CLASSES, hidden_channels=64).to(self.device)

        # Custom forward
        def custom_forward(x):
            y = []
            for i, m in enumerate(model.model):
                if hasattr(m, 'f') and m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if hasattr(m, 'i') and m.i in getattr(model, 'save', []) else None)
            p3 = y[4] if len(y) > 4 else None
            p4 = y[6] if len(y) > 6 else None
            p5 = y[8] if len(y) > 8 else None
            if p3 is None or p4 is None or p5 is None:
                return None
            p3 = model.lat_conv3(p3)
            p4 = model.lat_conv4(p4)
            p5 = model.lat_conv5(p5)
            features = model.bifpn_eca([p3, p4, p5])
            return model.lsf_detect(features)

        model.forward = custom_forward
        model.to(self.device)

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        print("All parameters are now trainable.")

        print("\nModel structure after modification:")
        for i, (name, module) in enumerate(model.named_children()):
            if i <= 15:
                print(f"{i}: {name} - {module.__class__.__name__}")

    def prepare_dataset(self):
        if self.dataset_prepared:
            return True
        print(f"Client {self.client_id} preparing dataset...")
        os.makedirs(self.train_images, exist_ok=True)
        os.makedirs(self.train_labels, exist_ok=True)
        os.makedirs(self.val_images, exist_ok=True)
        os.makedirs(self.val_labels, exist_ok=True)
        for folder in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        valid_images = []
        for label_file in os.listdir(self.annotations_path):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_path = os.path.join(self.images_path, base_name + ext)
                    if os.path.exists(img_path):
                        valid_images.append((base_name, ext))
                        break
        if os.path.exists(self.split_info_file):
            with open(self.split_info_file, 'r') as f:
                val_files = set(json.load(f))
        else:
            if len(valid_images) > 5:
                n_val = max(1, int(0.1 * len(valid_images)))
                val_indices = random.sample(range(len(valid_images)), n_val)
                val_files = {valid_images[i][0] for i in val_indices}
            else:
                val_files = set()
            with open(self.split_info_file, 'w') as f:
                json.dump(list(val_files), f)
        for base_name, ext in valid_images:
            img_src = os.path.join(self.images_path, base_name + ext)
            label_src = os.path.join(self.annotations_path, base_name + '.txt')
            if base_name in val_files:
                shutil.copy2(img_src, os.path.join(self.val_images, base_name + ext))
                shutil.copy2(label_src, os.path.join(self.val_labels, base_name + '.txt'))
            else:
                shutil.copy2(img_src, os.path.join(self.train_images, base_name + ext))
                shutil.copy2(label_src, os.path.join(self.train_labels, base_name + '.txt'))
        self.dataset_prepared = True
        return True

    def create_yaml(self):
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

    def local_train(self, epochs=10, project=None, name='train', round_idx=1):
        try:
            if not self.dataset_prepared:
                self.prepare_dataset()
            yaml_path = self.create_yaml()
            device_str = '0' if self.device.type == 'cuda' else 'cpu'
            # Only first round uses pretrained weights; after that use the global model
            use_pretrained = (round_idx == 1)
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
                workers=0,
                pretrained=use_pretrained   # critical fix
            )
            return True
        except Exception as e:
            print(f"Training failed: {e}")
            traceback.print_exc()
            return False

    def get_non_bn_weights(self):
        state_dict = self.model.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items()
                if not any(term in k for term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'])}

    def update_with_global_weights(self, global_weights):
        client_state = self.model.model.state_dict()
        for key in client_state:
            if not any(term in key for term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']):
                if key in global_weights and client_state[key].shape == global_weights[key].shape:
                    client_state[key] = global_weights[key].to(self.device)
        self.model.model.load_state_dict(client_state, strict=False)

    def connect_to_server(self, action='send_weights', round_idx=1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.settimeout(30000)
                sock.connect((self.server_ip, self.port))
                sock.sendall(f"CLIENT{self.client_id}".encode())
                if action == 'send_weights':
                    signal = sock.recv(5).decode()
                    if signal != "START":
                        return False
                    print(f"\n--- Round {round_idx} Local Training ---")
                    project = f"runs/client{self.client_id}"
                    if not self.local_train(epochs=10, project=project, name=f"round{round_idx}", round_idx=round_idx):
                        return False
                    non_bn_weights = self.get_non_bn_weights()
                    data = pickle.dumps(non_bn_weights)
                    sock.sendall(len(data).to_bytes(4, 'big'))
                    sock.sendall(data)
                    return True
                elif action == 'receive_model':
                    header = sock.recv(4)
                    if not header:
                        return False
                    msglen = int.from_bytes(header, 'big')
                    received = b''
                    while len(received) < msglen:
                        packet = sock.recv(min(4096, msglen - len(received)))
                        if not packet:
                            break
                        received += packet
                    if len(received) != msglen:
                        return False
                    buffer = io.BytesIO(received)
                    global_weights = torch.load(buffer, map_location='cpu')
                    self.update_with_global_weights(global_weights)
                    return True
            except Exception as e:
                print(f"Connection error: {e}")
                return False
        return False

    def run(self, rounds=5):
        self.prepare_dataset()
        self.create_yaml()
        for round_idx in range(1, rounds+1):
            print(f"\n=== Client {self.client_id} - Round {round_idx}/{rounds} ===")
            if not self.connect_to_server(action='send_weights', round_idx=round_idx):
                print(f"Failed to send weights in round {round_idx}")
                return
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
    client = SolarClient(client_id=client_id, base_path=base_path, server_ip='localhost', port=12000)
    client.run(rounds=7)