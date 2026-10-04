from ultralytics import YOLO
import torch.nn as nn
import torch
import torch.nn.functional as F

# Custom Modules Definitions
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

# Initialize YOLO model
model = YOLO("yolo11n.pt")

# Store original model structure
original_model = model.model.model

# Modify the model's structure
def modify_model(model):
    """Integrate custom neck and head into YOLO model."""
    # Create a new sequential module that includes:
    # 1. Backbone layers (original layers 0-8)
    # 2. Lateral convolution layers
    # 3. BiFPN with CBAM neck
    # 4. ASFF detection head
    
    # Extract backbone layers (layers 0-8)
    backbone_layers = nn.Sequential(*[original_model[i] for i in range(9)])
    
    # Add lateral convolution layers for feature reduction
    lat_conv3 = LateralConv(64, 128)
    lat_conv4 = LateralConv(128, 128)
    lat_conv5 = LateralConv(256, 128)

    # Integrate BiFPN with CBAM
    bifpn_cbam = BiFPN_CBAM(channels=128)

    # Replace detection head with ASFF
    asff_detect = ASFF_Detect(128, 80)  # Using default 80 classes for YOLO

    # Create a new model with the correct order
    class CustomYOLO(nn.Module):
        def __init__(self, backbone, lat_conv3, lat_conv4, lat_conv5, bifpn, head):
            super(CustomYOLO, self).__init__()
            self.backbone = backbone
            self.lat_conv3 = lat_conv3
            self.lat_conv4 = lat_conv4
            self.lat_conv5 = lat_conv5
            self.bifpn = bifpn
            self.head = head
            
        def forward(self, x):
            # Extract features from backbone
            y = []
            for i, m in enumerate(self.backbone):
                if hasattr(m, 'f') and m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x)
            
            # Get feature maps P3, P4, P5
            p3 = y[3]  # Output from layer 4
            p4 = y[5]  # Output from layer 6
            p5 = y[8]  # Output from layer 9

            # Apply lateral convolutions
            p3 = self.lat_conv3(p3)
            p4 = self.lat_conv4(p4)
            p5 = self.lat_conv5(p5)

            # Process through BiFPN-CBAM neck
            features = self.bifpn([p3, p4, p5])

            # Pass through ASFF detection head
            return self.head(features)
    
    # Replace the model with our custom architecture
    model.model = CustomYOLO(backbone_layers, lat_conv3, lat_conv4, lat_conv5, bifpn_cbam, asff_detect)

# Apply modifications
modify_model(model)

# Create a detailed summary function
def detailed_model_summary(model):
    """Print detailed model summary in the desired format."""
    print("from  n    params  module                                       arguments")
    
    # Track layer index
    layer_idx = 0
    
    # Print backbone layers
    for i, layer in enumerate(original_model[:9]):  # First 9 layers are backbone
        params = sum(p.numel() for p in layer.parameters())
        module_name = layer.__class__.__name__
        
        # Extract arguments based on layer type
        args = []
        try:
            if hasattr(layer, 'c1') and hasattr(layer.c1, 'in_channels'):
                # Conv layer
                args = [layer.c1.in_channels, layer.c1.out_channels, 
                       layer.c1.kernel_size[0] if hasattr(layer.c1.kernel_size, '__getitem__') else layer.c1.kernel_size,
                       layer.c1.stride[0] if hasattr(layer.c1.stride, '__getitem__') else layer.c1.stride]
            elif hasattr(layer, 'cv1') and hasattr(layer.cv1, 'conv'):
                # C3 or similar block
                args = [layer.cv1.conv.in_channels, layer.cv2.conv.out_channels]
                if hasattr(layer, 'n'):
                    args.append(layer.n)
                if hasattr(layer, 'shortcut'):
                    args.append(layer.shortcut)
        except Exception as e:
            # If we can't extract arguments, just continue without them
            pass
        
        args_str = f"[{', '.join(map(str, args))}]" if args else ""
        print(f"{layer_idx:3d}                  -1  1 {params:8d}  {module_name:40s} {args_str}")
        layer_idx += 1
    
    # Print lateral convolution layers
    lateral_layers = [
        ("LateralConv", model.model.lat_conv3, [64, 128]),
        ("LateralConv", model.model.lat_conv4, [128, 128]),
        ("LateralConv", model.model.lat_conv5, [256, 128])
    ]
    
    for name, layer, args in lateral_layers:
        params = sum(p.numel() for p in layer.parameters())
        args_str = f"[{', '.join(map(str, args))}]"
        print(f"{layer_idx:3d}                  -1  1 {params:8d}  {name:40s} {args_str}")
        layer_idx += 1
    
    # Print BiFPN_CBAM
    params = sum(p.numel() for p in model.model.bifpn.parameters())
    print(f"{layer_idx:3d}                  -1  1 {params:8d}  {'BiFPN_CBAM':40s} [128]")
    layer_idx += 1
    
    # Print ASFF_Detect
    params = sum(p.numel() for p in model.model.head.parameters())
    print(f"{layer_idx:3d}                  -1  1 {params:8d}  {'ASFF_Detect':40s} [128, 80]")

# Calculate total parameters and gradients
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the complete model summary
print("=" * 80)
print("COMPLETE MODEL SUMMARY WITH CUSTOM MODULES")
print("=" * 80)
detailed_model_summary(model)

# Print the final summary line
print(f"\nYOLO11n summary: {9 + 3 + 1 + 1} layers, {total_params:,} parameters, {trainable_params:,} gradients")
