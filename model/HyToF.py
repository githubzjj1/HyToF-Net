from typing import Type, Tuple, Optional, Set, List, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, Mlp, DropPath

from module.mgstformer import MGSTFormer
from module.mcl import MultiScaleSpatioTemporalFeatureConstructor, MotionConsistencyLearning
from module.contrastive import SpatialTemporalDecompose

from einops import rearrange

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod



class MFM_Layer(nn.Module):
    def __init__(
            self, A, depth, in_channels, out_channels, num_points=44, kernel_size=3, num_heads=4, num_edges=8, 
            drop=0., drop_path=0., mlp_ratio=2.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MFM_Layer, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                MGSTFormer(
                    A=A,
                    in_channels=in_channels if index == 0 else out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    num_edges=num_edges,
                    drop=drop,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input, dis_mask):
        output = input
        for block in self.blocks:
            output = block(output, dis_mask)
        return output
    
    def get_orthogonal_loss(self):
        loss = 0.0
        for block in self.blocks:
            if hasattr(block, 'get_orthogonal_loss'):
                loss += block.get_orthogonal_loss()
        return loss


class HyToF(nn.Module):
    def __init__(self, in_channels=2, depths=(3, 3, 3, 3), channels=(96, 96, 96, 96), num_classes=52, num_body_classes=6,
                 embed_dim=96, num_people=1, num_frames=64, num_points=44, kernel_size=3, num_heads=4, num_edges=8,
                 head_drop=0., drop=0., drop_path=0., mlp_ratio=2.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, index_t=True, global_pool='avg', graph=None, graph_args=dict(),
                 spatial_topk=12, semantic_topk=9):

        super(HyToF, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        
        A = self.graph.A
        self.part_mask = self.graph.part_mask
        self.num_part = self.graph.num_part


        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        self.num_body_classes = num_body_classes

        self.spatial_topk = spatial_topk
        self.semantic_topk = semantic_topk

        print(f'spatial_topk:{self.spatial_topk}')
        print(f'semantic_topk:{self.semantic_topk}')


        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=3 * in_channels, kernel_size=1, stride=1, padding=0),
            act_layer(),
            nn.Conv2d(in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=1, stride=1, padding=0)
        )

        if self.index_t:
            self.STPE = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.STPE, std=.02)
        else:
            self.STPE_ = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people))
            trunc_normal_(self.STPE_, std=.02)

        
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        MFM = []
        DS = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            MFM.append(
                MFM_Layer(
                    A=A,
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    num_edges=num_edges,
                    drop=drop,
                    drop_path=drop_path[sum(depths[:index]):sum(depths[:index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
            if index != len(depths) - 1:
                DS.append(
                    MultiScaleSpatioTemporalFeatureConstructor(channels[index], channels[index + 1], kernel_size=kernel_size)
                )
        self.MFM = nn.ModuleList(MFM)
        self.DSs = nn.ModuleList(DS)

        self.global_pool: str = global_pool

        self.head = nn.Linear(channels[-1], num_classes)
        self.head_body = nn.Linear(channels[-1], num_body_classes)

        self.fusion = MotionConsistencyLearning(in_channels=channels[-1])

        self.part_mlp = nn.Sequential(
            nn.Linear(channels[0], channels[0]//2),
            nn.ReLU(),
            nn.Linear(channels[0]//2, 1)
        )
        
        
        self.proj_dim = 128
        self.clb_proj_head = SpatialTemporalDecompose(in_channels=channels[-1], proj_dim=self.proj_dim)
    
    def part_score_mask(self, x, part_mask, mode='mean'):
        B, C, T, V = x.shape
        P = part_mask.shape[0]

        part_mask_tensor = torch.as_tensor(part_mask, dtype=torch.float32, device=x.device)

        x = rearrange(x, ' b c t v -> (b t) c v')

        mask = part_mask_tensor

        if mode == 'mean':
            num = mask.sum(dim=1, keepdim=True)  
            num[num==0] = 1
            mask = mask / num
        
        part = torch.einsum('b c v, p v -> b c p', x, mask)
        part = rearrange(part, 'b c p -> b p c')

        part_score = self.part_mlp(part)
        part_score = rearrange(part_score, '(b t) p 1 -> b t p 1', b=B, t=T).mean(dim=1)
        part_score = F.softmax(part_score, dim=1).squeeze(-1) 

        node_weight = torch.matmul(part_score, part_mask_tensor) 

        return node_weight


    def forward_features(self, X_feat, dis_mask):
        X_f_L = []
        DSs = [X_feat]

        ds_output = X_feat
        for ds in self.DSs:
            ds_output = ds(ds_output)
            DSs.append(ds_output)

        for X_f, mfm_layer in zip(DSs, self.MFM):
            X_f_L.append(mfm_layer(X_f, dis_mask))

        X_z = self.fusion(X_f_L)
        return X_z

    def classifier(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input)
    
    def classifier_body(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head_body(input)


    def feature_embedding(self, X_raw, index_t):
        B, C, T, V, M = X_raw.shape

        X_raw = X_raw.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  

        output = self.projection(X_raw)

        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)  
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            X_feat = output + torch.einsum('b t c, c v -> b c t v', te, self.STPE)
        else:
            X_feat = output + self.STPE_

        part_mask = self.part_score_mask(X_feat, self.part_mask) 
        
        return X_feat, part_mask

    def forward(self, input, index_t):
        X_feat, part_mask = self.feature_embedding(input, index_t)
        X_z = self.forward_features(X_feat, part_mask)
        
        logits= self.classifier(X_z)

        
        logits_body= self.classifier_body(X_z)

        if self.training:
            f_s, f_t = self.clb_proj_head(X_z)
            ortho_loss = self.get_orthogonal_loss()
            return logits, logits_body, f_s, f_t, ortho_loss

        return logits, logits_body
    
    def get_orthogonal_loss(self):
        loss = 0.0
        for mfm in self.MFM:
            loss += mfm.get_orthogonal_loss()
        return loss


def HyToF_(**kwargs):
    return HyToF(
        depths=(3, 3, 3, 3),
        channels=(96, 96, 96, 96),
        embed_dim=96,
        **kwargs
    )

