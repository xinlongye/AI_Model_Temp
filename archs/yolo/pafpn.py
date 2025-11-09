# -*- coding:utf-8 -*-
# @FileName  :pafpn.py
# @Time      :2025/10/12 10:30
# @Author    :yxl

import torch
from .cspdarknet import C2f, CBA


# yolov8
class PAFPNV8(torch.nn.Module):
    def __init__(self, net_depth=1, net_width=16, width_ratio=1.0):
        super().__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_p4 = C2f(in_ch=int(net_width * 16 * width_ratio) + net_width * 8,
                           out_ch=net_width * 8,
                           n=net_depth,
                           shortcut=False)

        self.conv_p3 = C2f(in_ch=net_width * 8 + net_width * 4,
                           out_ch=net_width * 4,
                           n=net_depth,
                           shortcut=False)

        self.downsample_p3 = CBA(in_ch=net_width * 4,
                                 out_ch=net_width * 4,
                                 kernel_size=3,
                                 stride=2)

        self.conv_p4_out = C2f(in_ch=net_width * 8 + net_width * 4,
                               out_ch=net_width * 8,
                               n=net_depth,
                               shortcut=False)

        self.downsample_p4 = CBA(in_ch=net_width * 8,
                                 out_ch=net_width * 8,
                                 kernel_size=3,
                                 stride=2)
        self.conv_p5_out = C2f(in_ch=int(net_width * 16 * width_ratio) + net_width * 8,
                               out_ch=int(net_width * 16 * width_ratio),
                               n=net_depth,
                               shortcut=False)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        feat1, feat2, feat3 = x
        # 上采样
        p5_upsample = self.upsample(feat3)
        p4 = torch.cat(tensors=(p5_upsample, feat2), dim=1)
        p4 = self.conv_p4(p4)

        p4_upsample = self.upsample(p4)
        p3 = torch.cat(tensors=(p4_upsample, feat1), dim=1)
        p3 = self.conv_p3(p3)

        # 下采样
        p3_upsample = self.downsample_p3(p3)
        p4 = torch.cat(tensors=(p3_upsample, p4), dim=1)
        p4 = self.conv_p4_out(p4)

        p4_downsample = self.downsample_p4(p4)
        p5 = torch.cat(tensors=(p4_downsample, feat3), dim=1)
        p5 = self.conv_p5_out(p5)

        return p3, p4, p5

if __name__ == "__main__":
    print(__file__)
