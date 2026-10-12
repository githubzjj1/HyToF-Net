import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicHypergraphConv(nn.Module):
    def __init__(self, in_channels, num_hyperedges=16):
        
        super(DynamicHypergraphConv, self).__init__()
        self.num_hyperedges = num_hyperedges
        
        self.H_generator = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, 1),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels // 2, num_hyperedges, 1)
        )
        
        self.W = nn.Parameter(torch.ones(num_hyperedges))

    def forward(self, x):
        
        B, C, T, V = x.shape
        
        x_pool = x.mean(dim=2)  
        
        H = self.H_generator(x_pool)  
        H = H.transpose(1, 2)         
        
        H = F.softmax(H, dim=-1)      
        
        W = torch.diag_embed(self.W).unsqueeze(0).expand(B, -1, -1)  
        
        D_e = H.sum(dim=1) + 1e-6  
        D_e_inv = torch.diag_embed(1.0 / D_e)  
        
        A_hyper = H @ W @ D_e_inv @ H.transpose(1, 2)  
        
        D_v_inv = 1.0 / (A_hyper.sum(dim=-1, keepdim=True) + 1e-6)
        A_hyper = A_hyper * D_v_inv  
        
        out = torch.einsum('b c t u, b v u -> b c t v', x, A_hyper)
        
        return out