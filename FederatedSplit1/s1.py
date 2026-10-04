# ==================== Federated Server (server.py) ====================
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

# ... [Custom Modules: ChannelAttention, SpatialAttention, CBAM, ASFF, BiFPN_Block, BiFPN_CBAM, LateralConv, C3, ASFF_Detect] ...
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
CLASS_NAMES =['hot_spot', 'scratch', 'no_electricity', 'black_border', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

class FederatedServer:
    def __init__(self, num_clients=2, port=12000):
        self.num_clients = num_clients
        self.port = port
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("Initializing global model...")
        
        # Load base YOLO model
        self.global_model = YOLO("yolo11n.pt")
        self.modify_model()
        print("Global model initialized with BiFPN-CBAM and ASFF!")
        
        # Create dummy dataset directory
        self.dummy_dir = "server_dummy_data"
        os.makedirs(self.dummy_dir, exist_ok=True)
        self.create_dummy_dataset()

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

    def modify_model(self):
        """Modify YOLO model with custom neck and head"""
        model = self.global_model.model
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
    server.start(rounds=5)