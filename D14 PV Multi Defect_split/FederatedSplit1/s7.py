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

# ========== NEW: Refine Block (per scale) ==========
class RefineBlock(nn.Module):
    """Lightweight per‑scale refinement with GhostConv and residual"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = GhostConv(channels, channels, kernel_size=3, stride=1)
        self.conv2 = GhostConv(channels, channels, kernel_size=3, stride=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x + identity

# ========== NEW: CBAM (single block) ==========
class CBAM(nn.Module):
    """Convolutional Block Attention Module (single block)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial))
        return x * spatial_att

# ========== Ghost‑based Decoupled Head ==========
class GhostDecoupledDetect(nn.Module):
    """Ghost-based decoupled detection head for one feature level."""
    def __init__(self, in_channels, num_classes, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2
        self.shared = GhostConv(in_channels, hidden_channels, kernel_size=1)
        self.cls = GhostConv(hidden_channels, num_classes, kernel_size=1)
        self.reg = GhostConv(hidden_channels, 4, kernel_size=1)

    def forward(self, x):
        x = self.shared(x)
        cls = self.cls(x)
        reg = self.reg(x)
        return torch.cat([cls, reg], dim=1)

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

# ================== Server ==================
CLASS_NAMES = ['hot_spot', 'scratch', 'no_electricity', 'black_border', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

def count_parameters(model):
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
        self.global_model = YOLO("yolo11n.pt")
        total_params, trainable_params = count_parameters(self.global_model.model)
        print(f"Original model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        self.modify_model()
        total_params, trainable_params = count_parameters(self.global_model.model)
        print(f"Modified model: Total params: {total_params:,}, Trainable: {trainable_params:,}")
        print("Global model initialized with MobileNetV3Block, ECA, Refine, CBAM, and Ghost Decoupled Head!")
        self.dummy_dir = "server_dummy_data"
        os.makedirs(self.dummy_dir, exist_ok=True)
        self.create_dummy_dataset()

    def inspect_yolo_layers(self):
        model = self.global_model.model
        print("\n" + "="*80)
        print("YOLOv11 MODEL STRUCTURE INSPECTION")
        print("="*80)
        print("\nBackbone Layers (0-8):")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i <= 8:
                print(f"{i}: {name} - {module.__class__.__name__}")
                if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    print(f"    in_channels: {module.in_channels}, out_channels: {module.out_channels}")
        print("\nNeck Layers (after 8):")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i > 8:
                print(f"{i}: {name} - {module.__class__.__name__}")
                if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                    print(f"    in_channels: {module.in_channels}, out_channels: {module.out_channels}")
        print("\nLooking for Detect head...")
        for name, module in model.named_children():
            if 'detect' in name.lower():
                print(f"Found Detect head: {name} - {module.__class__.__name__}")
        print("="*80 + "\n")

    def modify_model(self):
        model = self.global_model.model
        self.inspect_yolo_layers()
        model.nc = NUM_CLASSES
        model.names = CLASS_NAMES

        # Replace layer 8 with MobileNetV3Block
        if len(list(model.children())) > 8:
            layers = list(model.children())
            layers[8] = MobileNetV3Block(256, 256).to(self.device)
            model.model = nn.Sequential(*layers)

        # Remove original neck layers (9-24)
        print("Removing original neck layers...")
        layers = list(model.model.children())
        neck_start_idx = 9
        neck_end_idx = min(25, len(layers))
        for i in range(neck_start_idx, neck_end_idx):
            if i < len(layers):
                layers[i] = nn.Identity()
                print(f"  Replaced layer {i} with Identity")
        model.model = nn.Sequential(*layers)

        # Remove original Detect head
        print("Removing original Detect head...")
        for name, module in model.named_children():
            if 'detect' in name.lower():
                print(f"  Found and removing {name}")
                setattr(model, name, nn.Identity())

        # Add custom modules
        print("Adding custom modules...")
        CH = 128  # channel size for neck
        model.lat_conv3 = LateralGhostConv(64, CH).to(self.device)
        model.lat_conv4 = LateralGhostConv(128, CH).to(self.device)
        model.lat_conv5 = LateralGhostConv(256, CH).to(self.device)
        model.bifpn_eca = BiFPN_ECA(channels=CH).to(self.device)

        # NEW: refinement per scale
        model.refine3 = RefineBlock(CH).to(self.device)
        model.refine4 = RefineBlock(CH).to(self.device)
        model.refine5 = RefineBlock(CH).to(self.device)

        # NEW: shared CBAM
        model.cbam = CBAM(channels=CH, reduction=16).to(self.device)

        # Ghost decoupled head
        model.lsf_detect = GhostDecoupledDetectMulti(CH, NUM_CLASSES, hidden_channels=CH//2).to(self.device)

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
                print(f"Warning: Could not extract feature maps.")
                return None
            # Lateral connections
            p3 = model.lat_conv3(p3)
            p4 = model.lat_conv4(p4)
            p5 = model.lat_conv5(p5)

            # BiFPN-ECA
            p3, p4, p5 = model.bifpn_eca([p3, p4, p5])

            # Refine per scale
            p3 = model.refine3(p3)
            p4 = model.refine4(p4)
            p5 = model.refine5(p5)

            # Shared CBAM
            p3 = model.cbam(p3)
            p4 = model.cbam(p4)
            p5 = model.cbam(p5)

            # Detection head
            return model.lsf_detect([p3, p4, p5])

        model.forward = custom_forward
        model.to(self.device)

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
        print("All parameters are now trainable.")

        print("\nModel structure after modification:")
        print("-"*40)
        for i, (name, module) in enumerate(model.named_children()):
            if i <= 15:
                print(f"{i}: {name} - {module.__class__.__name__}")
        print("\nCustom modules added:")
        print(f"  - lat_conv3/4/5: LateralGhostConv")
        print(f"  - bifpn_eca: BiFPN_ECA")
        print(f"  - refine3/4/5: RefineBlock")
        print(f"  - cbam: CBAM")
        print(f"  - lsf_detect: GhostDecoupledDetectMulti\n")

    def create_dummy_dataset(self):
        train_img_dir = os.path.join(self.dummy_dir, 'train', 'images')
        train_lbl_dir = os.path.join(self.dummy_dir, 'train', 'labels')
        val_img_dir = os.path.join(self.dummy_dir, 'val', 'images')
        val_lbl_dir = os.path.join(self.dummy_dir, 'val', 'labels')
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)
        dummy_img = Image.new('RGB', (64, 64), color='black')
        dummy_img_path = os.path.join(train_img_dir, 'dummy.jpg')
        dummy_img.save(dummy_img_path)
        with open(os.path.join(train_lbl_dir, 'dummy.txt'), 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        shutil.copy(dummy_img_path, os.path.join(val_img_dir, 'dummy.jpg'))
        shutil.copy(os.path.join(train_lbl_dir, 'dummy.txt'), os.path.join(val_lbl_dir, 'dummy.txt'))
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
        all_keys = set(global_dict.keys())
        for weights in self.client_weights.values():
            all_keys.update(weights.keys())
        avg_weights = {}
        for key in all_keys:
            if any(bn_term in key for bn_term in ['bn', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']):
                continue
            weight_list = []
            for client_id in range(1, self.num_clients+1):
                if client_id in self.client_weights and key in self.client_weights[client_id]:
                    weight_list.append(self.client_weights[client_id][key].to(self.device))
            if weight_list:
                avg_weights[key] = torch.stack(weight_list, dim=0).mean(0)
        new_state_dict = global_dict.copy()
        for k, v in avg_weights.items():
            if k in new_state_dict and v.shape == new_state_dict[k].shape:
                new_state_dict[k] = v
        self.global_model.model.load_state_dict(new_state_dict, strict=False)
        print("Federated averaging completed!")

    def handle_client(self, conn, addr, round_idx):
        try:
            print(f"Connected to {addr}")
            client_id_str = conn.recv(7).decode()
            if not client_id_str.startswith("CLIENT"):
                return None
            client_id = int(client_id_str[6:])
            print(f"Received connection from client {client_id}")
            conn.sendall("START".encode())
            header = conn.recv(4)
            if not header:
                return None
            msglen = int.from_bytes(header, 'big')
            received = b''
            while len(received) < msglen:
                packet = conn.recv(min(4096, msglen - len(received)))
                if not packet:
                    break
                received += packet
            if len(received) != msglen:
                return None
            client_weights = pickle.loads(received)
            return client_id, client_weights
        except Exception as e:
            print(f"Error handling client: {e}")
            return None
        finally:
            conn.close()

    def send_global_model(self, conn):
        try:
            buffer = io.BytesIO()
            torch.save(self.global_model.model.state_dict(), buffer)
            global_model_bytes = buffer.getvalue()
            conn.sendall(len(global_model_bytes).to_bytes(4, 'big'))
            total_sent = 0
            while total_sent < len(global_model_bytes):
                chunk = global_model_bytes[total_sent:total_sent+4096]
                sent = conn.send(chunk)
                if sent == 0:
                    raise RuntimeError("Socket connection broken")
                total_sent += sent
            return True
        except Exception as e:
            print(f"Error sending global model: {e}")
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
                print(f"Waiting for {self.num_clients} clients to submit weights...")
                for client_num in range(self.num_clients):
                    conn, addr = server_socket.accept()
                    result = self.handle_client(conn, addr, round_idx)
                    if result:
                        client_id, weights = result
                        self.client_weights[client_id] = weights
                        self.remaining_clients.discard(client_id)
                if len(self.client_weights) > 0:
                    self.federated_averaging()
                else:
                    print("No client weights received, skipping aggregation")
                save_path = f'global_round_{round_idx}.pt'
                torch.save(self.global_model.model.state_dict(), save_path)
                print(f"Saved global weights to {save_path}")
                print("\nDistributing updated global model to clients...")
                clients_to_send = set(range(1, self.num_clients+1))
                while clients_to_send:
                    print(f"Waiting for {len(clients_to_send)} clients to receive model...")
                    conn, addr = server_socket.accept()
                    try:
                        client_id_str = conn.recv(7).decode()
                        if client_id_str.startswith("CLIENT"):
                            client_id = int(client_id_str[6:])
                            if client_id in clients_to_send:
                                print(f"Sending global model to client {client_id}")
                                if self.send_global_model(conn):
                                    clients_to_send.remove(client_id)
                    except Exception as e:
                        print(f"Error during model distribution: {e}")
                    finally:
                        conn.close()
                print(f"===== Round {round_idx} Completed =====")
        except Exception as e:
            print(f"Server error: {e}")
            traceback.print_exc()
        finally:
            shutil.rmtree(self.dummy_dir, ignore_errors=True)
            print("Server shutdown")

if __name__ == "__main__":
    server = FederatedServer(num_clients=2, port=12000)
    server.start(rounds=7)