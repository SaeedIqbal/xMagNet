import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# 1. Enhanced Dataset Class with Multi-Task Labels
#---------------------------------------------------
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, dataset_name="TCGA-BRCA", transform=None):
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        self.transform = transform
        
        # Validate dataset structure
        required_columns = ['img_id', 'magnification', 'label', 'grade', 'subtype']
        assert all(col in self.metadata.columns for col in required_columns), \
            "Metadata missing required columns"

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_info = self.metadata.iloc[idx]
        mag = img_info['magnification']
        img_path = os.path.join(self.root_dir, f"{mag}x", img_info['img_id'])
        
        # Load image and masks
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.root_dir, "masks", img_info['img_id'])
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return {
            'image': image,
            'mask': mask.squeeze(0).long(),  # [H, W]
            'grade': torch.tensor(img_info['grade'], dtype=torch.float32),
            'subtype': torch.tensor(img_info['subtype'], dtype=torch.long),
            'magnification': torch.tensor(mag, dtype=torch.float32)
        }

#---------------------------------------------------
# 2. Point-Wise Reformation Block (PRB) with Skip
#---------------------------------------------------
class PRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4*in_channels, 1)
        self.conv2 = nn.Conv2d(4*in_channels, out_channels, 1)
        self.adaptive_skip = nn.Conv2d(in_channels, out_channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable skip weight

    def forward(self, x):
        identity = x
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        return self.alpha * x + (1 - self.alpha) * self.adaptive_skip(identity)

#---------------------------------------------------
# 3. Multi-Task Decoder with Grad-CAM Integration
#---------------------------------------------------
class MultiTaskDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, num_subtypes):
        super().__init__()
        
        # Segmentation Head
        self.prb = PRB(in_channels, in_channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, 1)
        )
        
        # Grading Regression Head
        self.grade_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Subtyping Classification Head
        self.subtype_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, num_subtypes)
        )
        
        # Grad-CAM hooks
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad  # Save gradients for Grad-CAM

    def forward(self, x, return_features=False):
        # Register hook for last conv layer in segmentation head
        x.requires_grad_(True)
        x.register_hook(self.activations_hook)
        self.activations = x.detach()
        
        # Segmentation
        seg_features = self.prb(x)
        seg_out = self.seg_head(seg_features)
        
        # Grading and Subtyping
        grade_out = self.grade_head(x).squeeze(-1)
        subtype_out = self.subtype_head(x)
        
        if return_features:
            return seg_out, grade_out, subtype_out, x
        return seg_out, grade_out, subtype_out
    
    def get_gradcam(self, target_class):
        """
        Compute Grad-CAM heatmap for classification task
        :param target_class: Subtype class index for explanation
        """
        assert self.gradients is not None, "Must run forward first!"
        
        # Pool gradients across channels
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weight activations by gradients
        weighted_activations = self.activations * pooled_gradients
        
        # Average across channels and apply ReLU
        heatmap = torch.mean(weighted_activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize and resize to input size
        heatmap = F.interpolate(heatmap, scale_factor=16, mode='bilinear')
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.squeeze().cpu().numpy()

#---------------------------------------------------
# 4. Uncertainty-Weighted Multi-Task Loss
#---------------------------------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(3))  # For σ_{Dice}, σ_{CE}, σ_{MSE}
        
    def forward(self, seg_pred, grade_pred, subtype_pred, seg_true, grade_true, subtype_true):
        # Dice Loss
        dice_loss = 1 - (2 * (seg_pred.softmax(1) * seg_true).sum() + 1e-8) / \
                      (seg_pred.softmax(1).sum() + seg_true.sum() + 1e-8)
        
        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(subtype_pred, subtype_true)
        
        # MSE Loss
        mse_loss = F.mse_loss(grade_pred.squeeze(), grade_true)
        
        # Uncertainty-weighted loss
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        precision3 = torch.exp(-self.log_vars[2])
        
        total_loss = (precision1 * dice_loss + self.log_vars[0] +
                     precision2 * ce_loss + self.log_vars[1] +
                     precision3 * mse_loss + self.log_vars[2])
        
        return total_loss, {'Dice': dice_loss.item(), 
                           'CE': ce_loss.item(),
                           'MSE': mse_loss.item()}

#---------------------------------------------------
# 5. Training Loop with Grad-CAM Visualization
#---------------------------------------------------
if __name__ == "__main__":
    # Initialize components
    encoder = HybridViT_SDC_Encoder().to(device)
    decoder = MultiTaskDecoder(in_channels=512, num_classes=3, num_subtypes=4).to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()},
        {'params': criterion.parameters(), 'weight_decay': 0}
    ], lr=1e-4)
    
    # Dataset and transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = BreastCancerDataset(
        root_dir="/home/phd/dataset/BreastCancer/",
        dataset_name="TCGA-BRCA",
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training loop
    for epoch in range(10):
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            grades = batch['grade'].to(device)
            subtypes = batch['subtype'].to(device)
            mags = batch['magnification'].to(device)
            
            # Forward pass
            fused_features = encoder(images, mags)
            seg_pred, grade_pred, subtype_pred = decoder(fused_features)
            
            # Compute loss
            loss, loss_dict = criterion(seg_pred, grade_pred, subtype_pred,
                                       masks, grades, subtypes)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Generate Grad-CAM example
            if epoch % 2 == 0:
                with torch.no_grad():
                    decoder.eval()
                    _, _, _, features = decoder(fused_features, return_features=True)
                    heatmap = decoder.get_gradcam(target_class=1)  # Example for HER2+
                    
                    # Save visualization
                    img_np = images[0].cpu().numpy().transpose(1,2,0)
                    heatmap_np = cv2.resize(heatmap, (256,256))
                    heatmap_np = np.uint8(255 * heatmap_np)
                    heatmap_np = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(img_np, 0.5, heatmap_np, 0.5, 0)
                    cv2.imwrite(f"gradcam_epoch{epoch}.jpg", superimposed)
                    
                    decoder.train()
        
        print(f"Epoch {epoch+1} Losses: {loss_dict}")