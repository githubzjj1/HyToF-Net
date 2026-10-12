import torch
from torch import nn
import torch.nn.functional as F
import torch_dct as dct

from einops import rearrange

class CascadedMSK_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, reduction=4):
        super(CascadedMSK_TCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.branch_5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.branch_7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        
        mid_channels = max(out_channels // reduction, 16)
        
        self.scale_mlp = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels * 3, kernel_size=1, bias=False)
        )
        
        
        self.keyframe_attn = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x3 = self.branch_3(x)   
        x5 = self.branch_5(x3)  
        x7 = self.branch_7(x5)  
        
        U = x3 + x5 + x7
        
        S = U.mean(dim=(2, 3), keepdim=True)
        
        A = self.scale_mlp(S)  
        
        A = rearrange(A, 'b (s c) t v -> b s c t v', s=3)
        A = F.softmax(A, dim=1)
        
        V_feat = x3 * A[:, 0] + x5 * A[:, 1] + x7 * A[:, 2]
        
        
        temp_pool = x.mean(dim=3, keepdim=True) 
        
        
        keyframe_weight = self.keyframe_attn(temp_pool)
        out = V_feat * keyframe_weight
        
        return out



class TimeFreqModeling(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4):
        super(TimeFreqModeling, self).__init__()
        
        
        self.tconv = CascadedMSK_TCN(channels, channels, kernel_size=kernel_size, groups=num_heads)
        self.freq_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        )
        self.tf_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        x_time = self.tconv(x)  

        x_trans = x.transpose(2, 3) 
        
        x_freq_domain = dct.dct(x_trans)
        
        x_freq_filtered = self.freq_filter(x_freq_domain)
        
        x_freq_out = dct.idct(x_freq_filtered).transpose(2, 3)

        x_cat = torch.cat([x_time, x_freq_out], dim=1)  
        out = self.tf_fusion(x_cat)                     

        return out