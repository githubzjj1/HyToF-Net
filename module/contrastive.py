import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalDecompose(nn.Module):
    def __init__(self, in_channels, proj_dim=128):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, C, T, V = x.shape

        
        f_s = x.mean(dim=2, keepdim=True) 
        f_s = self.spatial_conv(f_s)
        
        f_s = f_s.mean(dim=3).squeeze(-1)
        f_s = F.normalize(f_s, dim=-1)

        
        f_t = x.mean(dim=3, keepdim=True) 
        f_t = self.temporal_conv(f_t)
        
        f_t = f_t.mean(dim=2).squeeze(-1)
        f_t = F.normalize(f_t, dim=-1)

        return f_s, f_t
    

class ContrastiveLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, momentum=0.9, temperature=0.07, alpha=1.0, beta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = temperature
        self.alpha = alpha 
        self.beta = beta   

        
        self.register_buffer("prototypes", torch.randn(num_classes, feature_dim))
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    @torch.no_grad()
    def update_prototypes(self, features, labels, preds):
        
        
        pred_labels = torch.argmax(preds, dim=1)
        
        correct_mask = (pred_labels == labels)
        
        if not correct_mask.any():
            return

        tp_features = features[correct_mask]
        tp_labels = labels[correct_mask]
        
        
        for c in tp_labels.unique():
            c_mask = (tp_labels == c)
            c_features = tp_features[c_mask].mean(dim=0)
            c_features = F.normalize(c_features, dim=-1)
            
            
            self.prototypes[c] = F.normalize(
                self.momentum * self.prototypes[c] + (1 - self.momentum) * c_features, 
                dim=-1
            )

    def forward(self, features, labels, preds):
     
        probs = F.softmax(preds, dim=1)
        
        
        batch_size = features.size(0)
        p_i_a = probs[torch.arange(batch_size), labels].view(-1, 1) 
        
        sim_matrix = torch.matmul(features, self.prototypes.T) / self.temperature
        
        dynamic_margin_pos = (1 - p_i_a) * self.alpha
        dynamic_margin_neg = (1 - p_i_a) * self.beta
        
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        
        adjusted_sim = sim_matrix - (one_hot_labels * dynamic_margin_pos) - ((1 - one_hot_labels) * dynamic_margin_neg)
        
        
        loss = F.cross_entropy(adjusted_sim, labels)
        
        return loss

