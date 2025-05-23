import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# 1. Dataset Class with Demographic Metadata
#---------------------------------------------------
class EthicalBreastCancerDataset(Dataset):
    def __init__(self, root_dir, subgroup_type="age", transform=None):
        """
        Args:
            root_dir: Path to dataset (/home/phd/dataset/BreastCancer/)
            subgroup_type: 'age', 'ethnicity', or 'institution'
        """
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        self.transform = transform
        self.subgroup_type = subgroup_type
        
        # Validate subgroup metadata
        assert subgroup_type in ['age', 'ethnicity', 'institution'], \
            "Invalid subgroup type"
        assert subgroup_type in self.metadata.columns, \
            f"Metadata missing {subgroup_type} column"
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_info = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, "images", img_info['img_id'])
        mask_path = os.path.join(self.root_dir, "masks", img_info['img_id'])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return {
            'image': image,
            'mask': mask.squeeze(0).long(),
            'subgroup': torch.tensor(img_info[self.subgroup_type], dtype=torch.long),
            'label': torch.tensor(img_info['label'], dtype=torch.long),
            'grade': torch.tensor(img_info['grade'], dtype=torch.float32)
        }

#---------------------------------------------------
# 2. Adversarial Debiasing Model Components
#---------------------------------------------------
class DemographicDiscriminator(nn.Module):
    """Predicts demographic subgroup from latent features"""
    def __init__(self, input_dim, num_subgroups):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_subgroups)
        
    def forward(self, z):
        return self.net(z)

class FairMagNetX(nn.Module):
    def __init__(self, num_subgroups, lambda_adv=0.5):
        super().__init__()
        # Shared encoder from previous work
        self.encoder = HybridViT_SDC_Encoder()  # From previous implementation
        self.task_head = nn.Linear(512, 2)  # Binary classification
        
        # Adversarial components
        self.discriminator = DemographicDiscriminator(512, num_subgroups)
        self.lambda_adv = lambda_adv
        
        # Information bottleneck parameters
        self.kl_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, mag):
        z = self.encoder(x, mag).mean(dim=[2,3])  # Global average pooling
        return self.task_head(z), z
    
    def mutual_info_loss(self, z, subgroups):
        """Estimate I(z; a) using variational approximation"""
        log_q = F.log_softmax(self.discriminator(z), dim=1)
        return F.nll_loss(log_q, subgroups)  # Cross-entropy ≈ upper bound
    
    def adversarial_loss(self, z, subgroups):
        """Adversarial min-max loss"""
        pred_subgroups = self.discriminator(z.detach())
        d_loss = F.cross_entropy(pred_subgroups, subgroups)
        
        # Feature-level invariance
        z_perm = z[torch.randperm(z.size(0))]
        mi_loss = self.mutual_info_loss(z, subgroups)
        
        return d_loss - self.lambda_adv * mi_loss

#---------------------------------------------------
# 3. Subgroup-Aware Contrastive Loss
#---------------------------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z, labels, subgroups):
        """
        z: latent features [batch_size, dim]
        labels: task labels [batch_size]
        subgroups: demographic subgroups [batch_size]
        """
        # Normalize features
        z = F.normalize(z, dim=1)
        
        # Generate positive/negative masks
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_subgroup = subgroups.unsqueeze(0) != subgroups.unsqueeze(1)
        positive_mask = same_label & diff_subgroup
        negative_mask = ~same_label
        
        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature
        
        # Positive logits
        positives = sim[positive_mask].reshape(z.size(0), -1)
        log_pos = -torch.log(torch.exp(positives).sum(dim=1))
        
        # Negative logits
        negatives = sim[negative_mask].reshape(z.size(0), -1)
        log_neg = torch.logsumexp(negatives, dim=1)
        
        return (log_pos + log_neg).mean()

#---------------------------------------------------
# 4. Fairness Metrics Computation
#---------------------------------------------------
def compute_fairness_metrics(model, dataloader, subgroup_bins):
    """Compute AUC and ECE for each subgroup"""
    model.eval()
    all_preds = []
    all_labels = []
    all_subgroups = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            mags = batch['magnification'].to(device)
            outputs, _ = model(images, mags)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            all_preds.append(probs.cpu())
            all_labels.append(batch['label'])
            all_subgroups.append(batch['subgroup'])
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    subgroups = torch.cat(all_subgroups).numpy()
    
    metrics = {}
    for subgroup in subgroup_bins:
        mask = (subgroups == subgroup)
        if mask.sum() == 0:
            continue
            
        # Compute AUC
        auc = roc_auc_score(labels[mask], preds[mask])
        
        # Compute ECE
        prob_true, prob_pred = calibration_curve(labels[mask], preds[mask], n_bins=10)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        metrics[subgroup] = {'AUC': auc, 'ECE': ece}
    
    # Calculate fairness gap
    aucs = [v['AUC'] for v in metrics.values()]
    mean_auc = np.mean(aucs)
    fairness_gap = np.max(np.abs(aucs - mean_auc))
    
    return metrics, fairness_gap

#---------------------------------------------------
# 5. Training Loop with Ethical Components
#---------------------------------------------------
def train_fair_model():
    # Initialize components
    dataset = EthicalBreastCancerDataset(
        "/home/phd/dataset/BreastCancer/", 
        subgroup_type="age"
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = FairMagNetX(num_subgroups=3).to(device)  # Age subgroups: <40, 40-60, >60
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    contrastive_loss = ContrastiveLoss()
    
    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            images = batch['image'].to(device)
            mags = batch['magnification'].to(device)
            labels = batch['label'].to(device)
            subgroups = batch['subgroup'].to(device)
            
            # Forward pass
            outputs, z = model(images, mags)
            task_loss = F.cross_entropy(outputs, labels)
            
            # Adversarial loss
            adv_loss = model.adversarial_loss(z, subgroups)
            
            # Contrastive loss
            cont_loss = contrastive_loss(z, labels, subgroups)
            
            # Total loss
            total_loss = task_loss + adv_loss + 0.1 * cont_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Compute fairness metrics
        if epoch % 10 == 0:
            metrics, fairness_gap = compute_fairness_metrics(
                model, dataloader, subgroup_bins=[0, 1, 2]
            )
            print(f"Epoch {epoch}: Fairness Gap={fairness_gap:.4f}")
            for sg, vals in metrics.items():
                print(f"  Subgroup {sg}: AUC={vals['AUC']:.3f}, ECE={vals['ECE']:.3f}")
    
    return model

#---------------------------------------------------
# 6. Example Usage
#---------------------------------------------------
if __name__ == "__main__":
    # Train fair model
    fair_model = train_fair_model()
    
    # Save model
    torch.save(fair_model.state_dict(), "magnet_x_fair.pth")