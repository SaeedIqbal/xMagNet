import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# 1. Custom Dataset Class
#---------------------------------------------------
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, dataset_name="TCGA-BRCA", transform=None):
        """
        Args:
            root_dir (str): Path to dataset (e.g., '/home/phd/dataset/BreastCancer/')
            dataset_name (str): Subdirectory name (e.g., 'TCGA-BRCA' or 'Camelyon16')
            transform (callable): Optional transforms
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_info = self.metadata.iloc[idx]
        mag = img_info['magnification']
        img_path = os.path.join(self.root_dir, f"{mag}x", img_info['img_id'])
        image = Image.open(img_path).convert('RGB')
        label = img_info['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), mag

#---------------------------------------------------
# 2. Vision Transformer (ViT) Module
#---------------------------------------------------
class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))  # Learnable positional encoding

        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
            num_layers=n_layers
        )

    def forward(self, x):
        # Input shape: [batch, channels, H, W]
        patches = self.patch_embed(x)  # [batch, d_model, n_patches_h, n_patches_w]
        patches = patches.flatten(2).permute(0, 2, 1)  # [batch, n_patches, d_model]

        # Add positional embeddings
        patches = patches + self.pos_embed[:, 1:(self.n_patches + 1)]
        
        # Transformer processing
        features = self.transformer(patches)
        return features  # [batch, n_patches, d_model]

#---------------------------------------------------
# 3. Separable Dilation Convolution (SDC) Module
#---------------------------------------------------
class SDC(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation_rates=[2, 4, 6]):
        super().__init__()
        self.dilation_rates = dilation_rates

        # Depthwise separable convolutions
        self.depthwise = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=d*r, dilation=d, groups=in_channels)
            for d in dilation_rates
        ])
        self.pointwise = nn.Conv2d(in_channels * len(dilation_rates), out_channels, kernel_size=1)

    def forward(self, x):
        features = []
        for conv in self.depthwise:
            features.append(conv(x))
        x = torch.cat(features, dim=1)
        x = self.pointwise(x)
        return x

#---------------------------------------------------
# 4. Magnification-Aware Gating (MAG)
#---------------------------------------------------
class MAG(nn.Module):
    def __init__(self, d_model=512, sdc_channels=64):
        super().__init__()
        # Learnable parameters
        self.W_g = nn.Linear(d_model + sdc_channels, 1)
        self.beta = nn.Parameter(torch.tensor(0.1))  # Temperature scaling factor

    def forward(self, F_vit, F_sdc, magnification):
        # Reshape and project features
        F_vit = F_vit.mean(dim=1)  # Global average over patches [batch, d_model]
        F_sdc = F_sdc.flatten(1)   # Flatten SDC features [batch, sdc_channels * H * W]
        F_sdc = F_sdc[:, :self.W_g.in_features - F_vit.size(1)]  # Truncate if needed

        # Concatenate and compute gating weights
        combined = torch.cat([F_vit, F_sdc], dim=1)
        temperature = torch.exp(-self.beta * magnification.unsqueeze(1))
        gate = torch.sigmoid(self.W_g(combined) / temperature)

        return gate

#---------------------------------------------------
# 5. Hybrid ViT-SDC Encoder
#---------------------------------------------------
class HybridViT_SDC_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViT()
        self.sdc = SDC()
        self.mag = MAG()

    def forward(self, x, magnification):
        # Process with ViT (low mag) and SDC (high mag)
        F_vit = self.vit(x)
        F_sdc = self.sdc(x)

        # Magnification-aware fusion
        gate = self.mag(F_vit, F_sdc, magnification)
        F_fused = gate * F_vit.mean(dim=1).unsqueeze(-1).unsqueeze(-1) + (1 - gate) * F_sdc

        return F_fused

#---------------------------------------------------
# 6. Training Setup & Dataset Loading
#---------------------------------------------------
if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = BreastCancerDataset(
        root_dir="/home/phd/dataset/BreastCancer/",
        dataset_name="TCGA-BRCA",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = HybridViT_SDC_Encoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):
        for images, labels, mag in dataloader:
            images, labels, mag = images.to(device), labels.to(device), mag.to(device)
            
            # Forward pass
            outputs = model(images, mag)
            loss = criterion(outputs, labels.float())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# 1. Dataset Class (Enhanced with Magnification Metadata)
#---------------------------------------------------
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, dataset_name="TCGA-BRCA", transform=None):
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_info = self.metadata.iloc[idx]
        mag = img_info['magnification']
        img_path = os.path.join(self.root_dir, f"{mag}x", img_info['img_id'])
        image = Image.open(img_path).convert('RGB')
        label = img_info['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(mag, dtype=torch.float32)

#---------------------------------------------------
# 2. ViT Module (With Attention Return for Theorem 1 Validation)
#---------------------------------------------------
class ViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        self.n_heads = n_heads

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        # Generate patches [batch, d_model, n_patches]
        patches = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        patches = patches + self.pos_embed[:, 1:(self.n_patches + 1)]
        
        # Self-attention with saved weights for validation
        output = self.transformer(patches)
        return output

#---------------------------------------------------
# 3. SDC Module (With Gradient Tracking for Theorem 1)
#---------------------------------------------------
class SDC(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, dilation_rates=[2, 4, 6]):
        super().__init__()
        self.depthwise = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=d*r, dilation=d, groups=in_channels)
            for d in dilation_rates
        ])
        self.pointwise = nn.Conv2d(in_channels * len(dilation_rates), out_channels, kernel_size=1)

    def forward(self, x):
        features = [conv(x) for conv in self.depthwise]
        return self.pointwise(torch.cat(features, dim=1))

#---------------------------------------------------
# 4. MAG Module (Revised Temperature Scaling)
#---------------------------------------------------
class MAG(nn.Module):
    def __init__(self, d_model=512, sdc_channels=64):
        super().__init__()
        self.W_g = nn.Linear(d_model + sdc_channels, 1)
        self.beta = nn.Parameter(torch.tensor(0.1))  # Temperature parameter

    def forward(self, F_vit, F_sdc, magnification):
        # Reshape features
        F_vit = F_vit.mean(dim=1)  # [batch, d_model]
        F_sdc = F_sdc.flatten(1)    # [batch, sdc_channels * H * W]

        # Temperature scaling (τ(m) = exp(-βm))
        temperature = torch.exp(-self.beta * magnification.unsqueeze(1))
        gate = torch.sigmoid(self.W_g(torch.cat([F_vit, F_sdc], dim=1)) / temperature)
        return gate

#---------------------------------------------------
# 5. Hybrid Encoder with Feature Norm Tracking
#---------------------------------------------------
class HybridViT_SDC_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViT()
        self.sdc = SDC()
        self.mag = MAG()

    def forward(self, x, magnification):
        F_vit = self.vit(x)
        F_sdc = self.sdc(x)
        
        # Compute feature norms for Theorem 1 validation
        self.vit_norm = torch.norm(F_vit, p=2, dim=(1, 2)).mean().item()
        self.sdc_norm = torch.norm(F_sdc, p=2, dim=(1, 2, 3)).mean().item()

        gate = self.mag(F_vit, F_sdc, magnification)
        F_fused = gate.view(-1, 1, 1) * F_vit.mean(dim=1, keepdim=True) + (1 - gate.view(-1, 1, 1)) * F_sdc
        
        # Track fused feature norm
        self.fused_norm = torch.norm(F_fused, p=2, dim=(1, 2, 3)).mean().item()
        return F_fused

#---------------------------------------------------
# 6. Gradient Stability Checker (Theorem 1 Validation)
#---------------------------------------------------
def monitor_gradient_norms(model):
    """Track gradient norms of F_fused w.r.t. input X"""
    gradients = []

    def hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].norm().item())

    handle = model.register_backward_hook(hook)
    return handle, gradients

#---------------------------------------------------
# 7. Training Loop with Mathematical Validation
#---------------------------------------------------
if __name__ == "__main__":
    # Initialize
    model = HybridViT_SDC_Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load dataset
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    dataset = BreastCancerDataset("/home/phd/dataset/BreastCancer/", "TCGA-BRCA", transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Attach gradient monitor
    handle, gradients = monitor_gradient_norms(model)

    for epoch in range(10):
        for images, labels, mag in dataloader:
            images, labels, mag = images.to(device), labels.to(device), mag.to(device)
            
            # Forward pass
            outputs = model(images, mag)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print norms (Theorem 1 validation)
            print(f"ViT Norm: {model.vit_norm:.2f}, SDC Norm: {model.sdc_norm:.2f}, Fused Norm: {model.fused_norm:.2f}")
            print(f"Gradient Norm: {np.mean(gradients):.4f}")

    handle.remove()  # Remove hook after training
'''