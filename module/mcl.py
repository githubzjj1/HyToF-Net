import torch
from torch import nn
import torch.nn.functional as F

class MultiScaleSpatioTemporalFeatureConstructor(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1),
                                   dilation=(dilation, 1), padding_mode='replicate')
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x




class MotionConsistencyLearning(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super(MotionConsistencyLearning, self).__init__()
 
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(8, 1), stride=(8, 1)),
            nn.BatchNorm2d(in_channels),  
            nn.ReLU(inplace=True)  
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 1), stride=(4, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(2, 1), stride=(2, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.f4_proc = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, kernel_size=1)
        )

        self.dropout = nn.Dropout(dropout)  

        self.res_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
      
        f1, f2, f3, f4 = x

        f1_ds = self.down1(f1)  
        f2_ds = self.down2(f2)  
        f3_ds = self.down3(f3)  
        f4_proc = self.f4_proc(f4)  

        fused_features = torch.cat([f1_ds, f2_ds, f3_ds, f4_proc], dim=1)  

        gate_weights = self.gate_conv(fused_features)
        gate_weights = F.softmax(gate_weights, dim=1)  

        fused_output = (gate_weights[:, 0:1, :, :] * f1_ds +
                        gate_weights[:, 1:2, :, :] * f2_ds +
                        gate_weights[:, 2:3, :, :] * f3_ds +
                        gate_weights[:, 3:4, :, :] * f4_proc)

        fused_output = self.dropout(fused_output)  

        
        identity = self.res_conv(f4_proc)
        output = fused_output + identity

        return output
    

