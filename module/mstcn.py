import torch
from torch import nn
import torch.nn.functional as F
import torch_dct as dct

from einops import rearrange

class TemporalDifferenceConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super(TemporalDifferenceConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        
        
        self.conv_center = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), 
                                     groups=groups, bias=False)
        
        
        self.conv_diff = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), 
                                   padding=(padding, 0), groups=groups, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        out_center = self.conv_center(x)
        
        out_conv = self.conv_diff(x)
        
        sum_weight = self.conv_diff.weight.sum(dim=2, keepdim=True)  
        
        compensation = F.conv2d(x, sum_weight, groups=self.groups)
        
        out_diff = out_conv - compensation
        
        
        out = self.bn(out_center + out_diff)
        
        return self.act(out)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        
        self.apply(weights_init)

    def forward(self, x):
        
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class TimeFreqModeling(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4):
        super(TimeFreqModeling, self).__init__()
        
        self.tconv = nn.Conv2d(
            in_channels=channels, 
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads
        )
        
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



class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        mid_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)

class TemporalSEBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4, reduction=4):
        super(TemporalSEBlock, self).__init__()
        
        
        self.tconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads
        )
        
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
        
        self.se = SEBlock(channels, reduction=reduction)

    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x
    

    
class SEBlockV2(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlockV2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        
        mid_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        
        y = self.sigmoid(avg_out + max_out)
        
        
        return x * y

class TemporalSEBlockV2(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4, reduction=4):
        super(TemporalSEBlockV2, self).__init__()
        
        self.tconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads
        )
        
        
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        
        
        self.se = SEBlockV2(channels, reduction=reduction)

    def forward(self, x):
        residual = x  
        
        out = self.tconv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.se(out)
        
        return out + residual  
    


class CascadedMSK_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, reduction=4):
        super(CascadedMSK_TCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.branch_5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        self.branch_7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=groups, bias=False),
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



class CascadedMSK_TCNV2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, reduction=4):
        super(CascadedMSK_TCNV2, self).__init__()
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


class CascadedMSK_TCNV3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, reduction=4):
        super(CascadedMSK_TCNV3, self).__init__()
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

    def forward(self, x):
        x3 = self.branch_3(x)   
        x5 = self.branch_5(x3)  
        x7 = self.branch_7(x5)  
        
        
        U = x3 + x5 + x7
        
        
        S = U.mean(dim=(2, 3), keepdim=True)
        
        
        A = self.scale_mlp(S)  
        
        
        A = rearrange(A, 'b (s c) t v -> b s c t v', s=3)
        A = F.softmax(A, dim=1)
        
        
        out = x3 * A[:, 0] + x5 * A[:, 1] + x7 * A[:, 2]
        
        return out


class TimeFreqModelingV2(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4):
        super(TimeFreqModelingV2, self).__init__()
        
        
        self.tconv = MultiScale_TemporalConv(channels, channels, kernel_size=kernel_size)
        
        
        
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


class TimeFreqModelingV3(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=4):
        super(TimeFreqModelingV3, self).__init__()
        
        self.tconv = CascadedMSK_TCNV3(channels, channels, kernel_size=kernel_size, groups=num_heads)
        
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
    
