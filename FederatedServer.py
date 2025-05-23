import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import copy
from PIL import Image
from sklearn.metrics import pairwise_distances

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------
# 1. Federated Dataset Class with Institution Splits
#---------------------------------------------------
class FederatedBreastCancerDataset(Dataset):
    def __init__(self, root_dir, institution_id, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root
            institution_id (int): Institution ID (0-5)
            transform (callable): Optional transforms
        """
        self.root_dir = os.path.join(root_dir, f"institution_{institution_id}")
        self.metadata = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        self.transform = transform
        
        # Validate institution split
        assert 0 <= institution_id <= 5, "6 institutions supported (0-5)"
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_info = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, "images", img_info['img_id'])
        mask_path = os.path.join(self.root_dir, "masks", img_info['img_id'])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return {
            'image': image,
            'mask': mask.squeeze(0).long(),
            'grade': torch.tensor(img_info['grade'], dtype=torch.float32),
            'subtype': torch.tensor(img_info['subtype'], dtype=torch.long),
            'institution': torch.tensor(img_info['institution'], dtype=torch.long)
        }

#---------------------------------------------------
# 2. Bayesian Model with Monte Carlo Dropout
#---------------------------------------------------
class BayesianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        return self.dropout(self.layer(x))

class MagNetXGlobal(nn.Module):
    def __init__(self, num_classes=3, num_subtypes=4):
        super().__init__()
        # Shared encoder (from previous implementation)
        self.encoder = HybridViT_SDC_Encoder()
        
        # Bayesian heads
        self.seg_head = BayesianLayer(512, num_classes)
        self.grade_head = BayesianLayer(512, 1)
        self.subtype_head = BayesianLayer(512, num_subtypes)
        
        # Dropout variance parameters
        self.sigma = nn.Parameter(torch.zeros(3))  # For each task
        
    def forward(self, x, mag):
        x = self.encoder(x, mag)
        seg = self.seg_head(x.mean(dim=[2,3]))
        grade = self.grade_head(x.mean(dim=[2,3]))
        subtype = self.subtype_head(x.mean(dim=[2,3]))
        return seg, grade, subtype
    
    def mc_dropout_forward(self, x, mag, T=50):
        """Monte Carlo dropout forward passes"""
        seg_preds, grade_preds, subtype_preds = [], [], []
        for _ in range(T):
            seg, grade, subtype = self.forward(x, mag)
            seg_preds.append(seg.softmax(1))
            grade_preds.append(grade.sigmoid())
            subtype_preds.append(subtype.softmax(1))
        return seg_preds, grade_preds, subtype_preds

#---------------------------------------------------
# 3. Optimal Transport Regularization
#---------------------------------------------------
def sinkhorn_loss(features_m, features_g, eps=0.1, max_iter=50):
    """
    Compute Sinkhorn divergence between client and global features
    """
    C = pairwise_distances(features_m, features_g, metric='euclidean')
    K = torch.exp(-C / eps)
    
    # Initialize dual vectors
    u = torch.ones(features_m.size(0), device=device) / features_m.size(0)
    v = torch.ones(features_g.size(0), device=device) / features_g.size(0)
    
    for _ in range(max_iter):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)
        
    P = torch.diag(u) @ K @ torch.diag(v)
    return torch.sum(P * C) - eps * (torch.sum(P * torch.log(P + 1e-8)) - 1)

#---------------------------------------------------
# 4. Federated Client Class
#---------------------------------------------------
class FederatedClient:
    def __init__(self, client_id, dataset, lr=1e-4):
        self.client_id = client_id
        self.model = MagNetXGlobal().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
    def local_update(self, global_model, lambda_ot=0.1, lambda_reg=0.01):
        """Perform one local update step"""
        self.model.load_state_dict(global_model.state_dict())
        
        total_loss = 0
        for batch in self.loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            grades = batch['grade'].to(device)
            subtypes = batch['subtype'].to(device)
            mags = batch['magnification'].to(device)
            
            # Forward pass
            seg_pred, grade_pred, subtype_pred = self.model(images, mags)
            
            # Task losses
            seg_loss = F.cross_entropy(seg_pred, masks)
            grade_loss = F.mse_loss(grade_pred.squeeze(), grades)
            subtype_loss = F.cross_entropy(subtype_pred, subtypes)
            
            # Optimal transport regularization
            with torch.no_grad():
                global_features = global_model.encoder(images, mags).mean(dim=[2,3])
            client_features = self.model.encoder(images, mags).mean(dim=[2,3])
            ot_loss = sinkhorn_loss(client_features, global_features)
            
            # Total loss
            loss = (seg_loss + grade_loss + subtype_loss +
                   lambda_ot * ot_loss +
                   lambda_reg * torch.norm(list(self.model.parameters()) - 
                                          list(global_model.parameters())))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.loader), self.model.state_dict()

#---------------------------------------------------
# 5. Federated Server with Momentum Aggregation
#---------------------------------------------------
class FederatedServer:
    def __init__(self, num_institutions=6):
        self.global_model = MagNetXGlobal().to(device)
        self.momentum_buffer = copy.deepcopy(self.global_model.state_dict())
        self.beta = 0.9  # Momentum coefficient
        
    def aggregate(self, client_updates, client_sizes):
        """Momentum-enhanced FedAvg aggregation"""
        total_size = sum(client_sizes)
        weights = [size/total_size for size in client_sizes]
        
        # Compute weighted average
        avg_update = {}
        for key in self.global_model.state_dict().keys():
            avg_update[key] = sum(w * client_updates[i][key] 
                              for i, w in enumerate(weights))
            
        # Apply momentum
        for key in self.momentum_buffer:
            self.momentum_buffer[key] = (self.beta * self.momentum_buffer[key] +
                                      (1 - self.beta) * avg_update[key])
            
        # Update global model
        current_state = self.global_model.state_dict()
        new_state = {k: current_state[k] + self.momentum_buffer[k] 
                    for k in current_state}
        self.global_model.load_state_dict(new_state)
        
    def estimate_uncertainty(self, dataset, T=50):
        """Compute epistemic uncertainty on validation set"""
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        self.global_model.eval()
        
        uncertainties = []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                mags = batch['magnification'].to(device)
                
                # MC dropout forward passes
                seg_preds, _, _ = self.global_model.mc_dropout_forward(images, mags, T)
                seg_probs = torch.stack(seg_preds)
                
                # Compute variance
                var = seg_probs.var(dim=0).mean(dim=[1,2])
                uncertainties.extend(var.cpu().numpy())
                
        return np.mean(uncertainties)

#---------------------------------------------------
# 6. Training Loop with Uncertainty Quantification
#---------------------------------------------------
if __name__ == "__main__":
    # Initialize server and clients
    server = FederatedServer()
    clients = [
        FederatedClient(i, FederatedBreastCancerDataset(
            "/home/phd/dataset/BreastCancer/", institution_id=i))
        for i in range(6)
    ]
    
    # Simulated validation set (use one institution)
    val_dataset = FederatedBreastCancerDataset(
        "/home/phd/dataset/BreastCancer/", institution_id=0)
    
    # Federated training
    for round in range(50):
        # Client updates
        client_losses = []
        client_updates = []
        client_sizes = []
        
        for client in clients:
            loss, update = client.local_update(server.global_model)
            client_losses.append(loss)
            client_updates.append(update)
            client_sizes.append(len(client.loader.dataset))
            
        # Server aggregation
        server.aggregate(client_updates, client_sizes)
        
        # Estimate uncertainty
        uncertainty = server.estimate_uncertainty(val_dataset)
        
        print(f"Round {round+1}: "
              f"Avg Loss {np.mean(client_losses):.4f}, "
              f"Uncertainty {uncertainty:.4f}")
    
    # Save final model
    torch.save(server.global_model.state_dict(), "magnex_x_federated.pth")