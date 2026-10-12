import torch
import torch.nn as nn
import torch_dct as dct
import torch.nn.functional as F

from einops import rearrange


class DualStreamMotion(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels

        self.time_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, 
                      padding=(1, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        init_mask = torch.linspace(2.0, -2.0, seq_len)
        self.freq_mask = nn.Parameter(init_mask.view(1, 1, seq_len, 1))

        self.low_freq_processor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.high_freq_processor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.freq_smooth = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False), 
            nn.Softmax(dim=1) 
        )

    def forward(self, x_delta):
        B, C, T, V = x_delta.shape

        out_time = self.time_branch(x_delta)

        x_perm = rearrange(x_delta, 'b c t v -> b c v t')
        x_freq_raw = dct.dct(x_perm, norm='ortho')
        
        x_freq = rearrange(x_freq_raw, 'b c v t -> b c t v')

        
        current_mask = self.freq_mask 
        if T != self.seq_len:
            mask_tmp = rearrange(current_mask, '1 1 s 1 -> 1 1 s')
            mask_tmp = F.interpolate(mask_tmp, size=T, mode='linear', align_corners=True)
            current_mask = rearrange(mask_tmp, '1 1 t -> 1 1 t 1')

        low_mask = torch.sigmoid(current_mask)
        high_mask = 1.0 - low_mask
        
        out_low = self.low_freq_processor(x_freq * low_mask)
        out_high = self.high_freq_processor(x_freq * high_mask)
        x_freq_fused = out_low + out_high

        
        
        x_fused_perm = rearrange(x_freq_fused, 'b c t v -> b c v t')
        x_time_recon_raw = dct.idct(x_fused_perm, norm='ortho')
        
        x_time_recon = rearrange(x_time_recon_raw, 'b c v t -> b c t v')
        
        out_freq = self.freq_smooth(x_time_recon)
        
        combined_features = torch.cat([out_time, out_freq], dim=1) 
        gate_weights = self.fusion_gate(combined_features)         

        
        weight_time = gate_weights[:, 0:1, :, :]
        weight_freq = gate_weights[:, 1:2, :, :]

        
        out = (out_time * weight_time) + (out_freq * weight_freq)
        
        return out + x_delta
    




class GraphConvGuidance(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GraphConvGuidance, self).__init__()

        self.adj = nn.Parameter(torch.eye(num_nodes))
        
        self.fc = nn.Linear(in_channels, out_channels)
        
        self.tanh = nn.Tanh()
        
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1).contiguous()  
        
        x = torch.einsum('vw,btwc->btwc', self.adj, x)  
        
        x = self.fc(x)  
        x = self.tanh(x)
        
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn(x)

        return x