import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from engine.extre_module.custom_nn.featurefusion.cgfm import ContextGuideFusionModule
from engine.deim.hybrid_encoder import HybridEncoder
from ..core import register
__all__ = ['HybridEncoder_CGFM']


@register()
class HybridEncoder_CGFM(HybridEncoder):
    __share__ = ['eval_spatial_size', ]
    
    def __init__(self, in_channels=..., feat_strides=..., hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0, enc_act='gelu', use_encoder_idx=..., num_encoder_layers=1, pe_temperature=10000, expansion=1, depth_mult=1, act='silu', eval_spatial_size=None, version='dfine'):
        super().__init__(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion, depth_mult, act, eval_spatial_size, version)
        # fpn
        self.fpn_feat_fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.fpn_feat_fusion_blocks.append(
                ContextGuideFusionModule([hidden_dim, hidden_dim], hidden_dim*2)
            )
            
        # pan
        self.pan_feat_fusion_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.pan_feat_fusion_blocks.append(
                ContextGuideFusionModule([hidden_dim, hidden_dim], hidden_dim*2)
            )
            
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 此处通道全部是hidden_dim
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            # 通道数是 hidden_dim
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # 把通道拼接换成fusion_blocks
            # inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](
                #                                            hidden_dim 参数在构造的时候已经确定了, 故此处无需再次传入
                self.fpn_feat_fusion_blocks[len(self.in_channels)-1-idx]([upsample_feat, feat_low])
                )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 通道数是 hidden_dim*2
            # [batch, channel, height, width]  dim=1 按照channel维度拼接！
            # 把通道拼接换成fusion_blocks
            # out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            out = self.pan_blocks[idx](
                self.pan_feat_fusion_blocks[idx]([downsample_feat, feat_height])
            )
            outs.append(out)

        return outs
        
    