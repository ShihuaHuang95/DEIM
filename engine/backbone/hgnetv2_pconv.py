"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
from .hgnetv2 import HG_Block, HG_Stage, HGNetv2
from engine.extre_module.custom_nn.conv_module.pconv import Partial_Conv

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

class HG_Block_PConv(HG_Block):
    def __init__(self, in_chs, mid_chs, out_chs, layer_num, kernel_size=3, residual=False, light_block=False, use_lab=False, agg='ese', drop_path=0):
        super().__init__(in_chs, mid_chs, out_chs, layer_num, kernel_size, residual, light_block, use_lab, agg, drop_path)

        if light_block:
            self.layers = nn.ModuleList()
            for i in range(layer_num):
                self.layers.append(
                    Partial_Conv(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                    )
                )

        


class HG_Stage_PConv(HG_Stage):
    def __init__(self, in_chs, mid_chs, out_chs, block_num, layer_num, downsample=True, light_block=False, kernel_size=3, use_lab=False, agg='ese', drop_path=0):
        super().__init__(in_chs, mid_chs, out_chs, block_num, layer_num, downsample, light_block, kernel_size, use_lab, agg, drop_path)
        
        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block_PConv(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)



@register()
class HGNetv2_PConv(HGNetv2):
    def __init__(self, name, use_lab=False, return_idx=..., freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True, agg='se', local_model_dir='weight/hgnetv2/'):
        super().__init__(name, use_lab, return_idx, freeze_stem_only, freeze_at, freeze_norm, pretrained, agg, local_model_dir)
        
        stage_config = self.arch_configs[name]['stage_config']

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage_PConv(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    agg))

        '''
        注意此处修改逻辑：
        如果把“冻结逻辑”放在父类的 __init__ 里，父类根本就不知道子类会在后面添加哪些额外模块，
        也不知道“要不要冻结这些子模块”。唯一能做的就是在父类里把自己那一套层（stem、backbone 的通用部分、某些归一化层等）先冻结，
        而对“子类专有的阶段”无能为力。 只有在“父类部分 + 子类部分的所有模块都已经创建完毕”之后，再去“遍历并冻结它们的参数”，才能保证冻结操作对所有目标层都生效。
        '''
        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

