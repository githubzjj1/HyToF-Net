import torch
from torch import nn
import torch.nn.functional as F

from timm.layers import trunc_normal_, Mlp, DropPath

from einops import rearrange

from torch.autograd import Variable

from module.motion import GraphConvGuidance
from module.tcn import TimeFreqModeling

class MGSTFormer(nn.Module):
    def __init__(self, A=None, in_channels=2, num_points=44, kernel_size=3, num_heads=4,
                 num_edges=8, drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MGSTFormer, self).__init__()

        self.num_edges = num_edges
        self.num_heads = num_heads

        self.mapping = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.norm_1 = norm_layer(in_channels)

        self.gconv = nn.Parameter(torch.zeros(num_heads, num_points, num_points))
        trunc_normal_(self.gconv, std=.02)

        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        self.alpha_part = nn.Parameter(torch.ones(1))
        self.alpha_hyper = nn.Parameter(torch.zeros(1))

        self.temporal_compress = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(3, 1), padding=(1, 0), groups=in_channels // 4),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, num_points)) 
        )

        
        self.joint_mapping = nn.Sequential(
            nn.Conv1d(in_channels // 4, num_heads * in_channels // 4, kernel_size=1, groups=num_heads),
            nn.LeakyReLU(),
            nn.Conv1d(num_heads * in_channels // 4, num_heads, kernel_size=1),
            nn.Tanh()
        )

        
        c_head = (in_channels // 4) // num_heads
        self.prototypes = nn.Parameter(torch.randn(num_heads, num_edges, c_head))
        trunc_normal_(self.prototypes, std=.02)

        self.edge_mapping = nn.Sequential(
            nn.Conv1d(in_channels // 4, num_heads * in_channels // 4, kernel_size=1, groups=num_heads),
            nn.LeakyReLU(),
            nn.Conv1d(num_heads * in_channels // 4, num_heads, kernel_size=1),
            nn.Sigmoid()
        )

        
        
        self.tconv = TimeFreqModeling(in_channels // 4, kernel_size=kernel_size, num_heads=num_heads)

        
        self.Motion = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2,
                      kernel_size=(3, 1), stride=1,
                      padding=(1, 0), groups=in_channels // 2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU()
        )

        
        self.proj = nn.Linear(in_channels, in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(in_features=in_channels,
                       hidden_features=int(mlp_ratio * in_channels),
                       act_layer=act_layer, drop=drop)

        self.MTM = nn.Sequential(
            nn.Conv2d(in_channels // 2, (in_channels // 4) * 2,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.BatchNorm2d((in_channels // 4) * 2)
        )

        self.MSM = GraphConvGuidance(in_channels // 2, (in_channels // 4) * 2, num_points)

        self.aggregation_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        
        self.aggregation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def interpolate_diff(self, x, order=0, dim=2):
        
        out = x
        for _ in range(order):
            out = torch.diff(out, dim=dim)
            
            out = F.pad(out, (0, 0, 0, 1), mode='replicate')
        return out

    def hyper_norm(self, H, W):
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=2, keepdim=True) + 1e-8
        w_ = w / norm_w

        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        A = h_ @ w_ @ H.transpose(3, 2)
        return A
    
    def part_graph(self, x, part_mask):
        B, N, C, T, V = x.shape

        
        x = x.mean(dim=3) 
        x1 = rearrange(x, 'b n c v -> b (n c) v')

        
        joint_weight = self.joint_mapping(x1)

        attn = torch.einsum('b n c v, b n c u -> b n v u', x, x) 

        mask_row = part_mask.unsqueeze(1).unsqueeze(-1) 
        mask_col = part_mask.unsqueeze(1).unsqueeze(-2) 
        attn = attn * mask_row * mask_col 

        G = self.hyper_norm(attn, joint_weight)

        return G
    
    def hypergraph(self, x, tau=1.0):
        B, N_heads, C_h, T, V = x.shape
        
        x_compress = rearrange(x, 'b n c t v -> b (n c) t v')
        x_spatial = self.temporal_compress(x_compress) 
        x_spatial = rearrange(x_spatial, 'b (n c) 1 v -> b n c v', n=N_heads)

        sim = torch.einsum('b n c v, n e c -> b n v e', x_spatial, self.prototypes)
        sim = sim / (C_h ** 0.5)
        H = F.gumbel_softmax(sim, tau=tau, hard=True, dim=-1) 

        hyperedge_feat = torch.einsum('b n c v, b n v e -> b n c e', x_spatial, H)
        hyperedge_feat_flat = rearrange(hyperedge_feat, 'b n c e -> b (n c) e')
        W = self.edge_mapping(hyperedge_feat_flat)
        W_mat = torch.diag_embed(W) 

        HW = torch.matmul(H, W_mat)
        D_v = HW.sum(dim=-1)
        D_v_inv_sqrt = torch.clamp(D_v, min=1e-6) ** (-0.5)
        D_v_mat = torch.diag_embed(D_v_inv_sqrt) 

        D_e = H.sum(dim=2)
        D_e_inv = torch.clamp(D_e, min=1e-6) ** (-1.0)
        D_e_mat = torch.diag_embed(D_e_inv) 

        temp = torch.matmul(HW, D_e_mat)
        temp = torch.matmul(temp, H.transpose(-1, -2))
        G = torch.matmul(D_v_mat, temp)
        G = torch.matmul(G, D_v_mat)

        G_norm = F.normalize(G, p=1, dim=-1)

        return G_norm
    
    def get_orthogonal_loss(self):
        P = self.prototypes 
        P_norm = F.normalize(P, p=2, dim=-1)
        
        gram = torch.bmm(P_norm, P_norm.transpose(1, 2)) 
        
        I = torch.eye(self.num_edges, device=P.device).unsqueeze(0).expand(self.num_heads, -1, -1)
        
        ortho_loss = F.mse_loss(gram, I)
        return ortho_loss


    def forward(self, X_feat, dis_mask):
      
        B, C, T, V = X_feat.shape

        
        X_feat = X_feat.permute(0, 2, 3, 1).contiguous()  
        X_feat_res = X_feat

        X_in = self.mapping(self.norm_1(X_feat)).permute(0, 3, 1, 2).contiguous()  
        f_st, X_in_motion = torch.split(X_in, [C // 2, C // 2], dim=1)
        X_in_tc = torch.chunk(f_st, 2, dim=1)  

        
        X_gc_ = []
        X_in_gc = torch.chunk(X_in_tc[0], self.gconv.shape[0], dim=1)
        X_in_gc_tensor = torch.stack(X_in_gc, dim=1)

        G_part = self.part_graph(X_in_gc_tensor, dis_mask)
        G_hyper = self.hypergraph(X_in_gc_tensor, tau=1.0)

        for i in range(self.gconv.shape[0]):
            Graph_total = self.gconv[i].unsqueeze(0) + self.A.unsqueeze(0) + \
                          self.alpha_part * G_part[:, i] + \
                          self.alpha_hyper * G_hyper[:, i]
            z = torch.einsum('n c t u, n v u -> n c t v', X_in_gc[i], Graph_total)
            X_gc_.append(z)
        X_gc = torch.cat(X_gc_, dim=1)  

        
        X_tc = self.tconv(X_in_tc[1])  

        
        X_delta = self.interpolate_diff(X_in_motion, order=1)
        motion = self.Motion(X_delta)  

        
        Z_s = self.MSM(motion)
        gamma_s, beta_s = torch.chunk(Z_s, 2, dim=1)

        mean_g = X_gc.mean(dim=(2, 3), keepdim=True)
        std_g = X_gc.std(dim=(2, 3), keepdim=True)
        X_gcm = (X_gc - mean_g) / (std_g + 1e-6) * (1 + gamma_s) + beta_s

        
        Z_t = self.MTM(motion)
        gamma_t, beta_t = torch.chunk(Z_t, 2, dim=1)

        mean_t = X_tc.mean(dim=(2, 3), keepdim=True)
        std_t = X_tc.std(dim=(2, 3), keepdim=True)
        X_tcm = (X_tc - mean_t) / (std_t + 1e-6) * (1 + gamma_t) + beta_t

        
        X_agg = torch.cat([X_tcm, X_gcm, motion], dim=1)  
        agg_weights = self.aggregation_gate(X_agg)  
        weighted_t = X_tc * agg_weights[:, 0:1, :, :]
        weighted_g = X_gc * agg_weights[:, 1:2, :, :]
        weighted_d = motion * agg_weights[:, 2:3, :, :]
        agg_feature = torch.cat([weighted_t, weighted_g, weighted_d], dim=1)
        agg_feature = self.aggregation(agg_feature)

        
        X_f = self.proj_drop(self.proj(agg_feature.permute(0, 2, 3, 1).contiguous()))
        X_f = X_feat_res + self.drop_path(X_f)

        X_f = X_f + self.drop_path(self.mlp(self.norm_2(X_f)))
        X_f = X_f.permute(0, 3, 1, 2).contiguous()

        return X_f
