# -*- coding:utf-8 -*-
# @FileName  :yolov8_arch.py
# @Time      :2025/10/8 13:53
# @Author    :yxl

import torch
from yolo.cspdarknet import CSPDarkNetV8, CBA
from yolo.pafpn import PAFPNV8
import torchinfo
from model_utils.registry import ARCH_REGISTRY

def make_anchors(feat_shapes, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = feat_shapes[i]
        sx = torch.arange(end=w) + grid_cell_offset  # shift x
        sy = torch.arange(end=h) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    anchor_out = torch.cat(anchor_points).detach()
    stride_out = torch.cat(stride_tensor).detach()

    return anchor_out, stride_out


class DFL(torch.nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.conv = torch.nn.Conv2d(in_channels=reg_max,
                                    out_channels=1,
                                    kernel_size=1,
                                    stride=1,
                                    bias=False)
        self.conv.requires_grad_(False)
        conv_weight = torch.arange(reg_max, dtype=torch.float)
        conv_weight = conv_weight.view(1, reg_max, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(conv_weight)

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.reg_max, a).transpose(2, 1).softmax(1)).view(b, 4, a)

@ARCH_REGISTRY.register()
class YOLOV8Net(torch.nn.Module):
    def __init__(self, model_type="s", num_class=80, input_shape=(640, 640)):
        """
        :param model_type: 模型类型：n, s, m, l, x
        :param num_class: 种类数
        """
        super().__init__()
        self.num_classes = num_class
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00}
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        width_ratio_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50}

        dep_mul = depth_dict[model_type]  # 模型深度
        wid_mul = width_dict[model_type]  # 模型宽度
        width_ratio_mul = width_ratio_dict[model_type]  # 宽度比例

        base_depth = max(round(dep_mul * 3), 1)
        base_channels = int(wid_mul * 64)

        self.backbone = CSPDarkNetV8(net_depth=base_depth,
                                     net_width=base_channels,
                                     width_ratio=width_ratio_mul)

        self.neck = PAFPNV8(net_depth=base_depth,
                            net_width=base_channels,
                            width_ratio=width_ratio_mul)

        # yolov8 head
        self.stride = [8, 16, 32]
        self.reg_max = 16
        in_channels = [base_channels * 4, base_channels * 8, int(base_channels * 16 * width_ratio_mul)]
        print(in_channels)
        reg_out_channels = max((16, in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(in_channels[0], num_class)

        # 80 × 80
        self.p3_reg = torch.nn.Sequential(CBA(in_ch=in_channels[0],
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=reg_out_channels,
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=reg_out_channels,
                                                          out_channels=4 * self.reg_max,
                                                          kernel_size=1))

        self.p3_cls = torch.nn.Sequential(CBA(in_ch=in_channels[0],
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=cls_out_channels,
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=cls_out_channels,
                                                          out_channels=num_class,
                                                          kernel_size=1))

        # 40 × 40
        self.p4_reg = torch.nn.Sequential(CBA(in_ch=in_channels[1],
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=reg_out_channels,
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=reg_out_channels,
                                                          out_channels=4 * self.reg_max,
                                                          kernel_size=1))

        self.p4_cls = torch.nn.Sequential(CBA(in_ch=in_channels[1],
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=cls_out_channels,
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=cls_out_channels,
                                                          out_channels=num_class,
                                                          kernel_size=1))

        # 80 × 80
        self.p5_reg = torch.nn.Sequential(CBA(in_ch=in_channels[2],
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=reg_out_channels,
                                              out_ch=reg_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=reg_out_channels,
                                                          out_channels=4 * self.reg_max,
                                                          kernel_size=1))

        self.p5_cls = torch.nn.Sequential(CBA(in_ch=in_channels[2],
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          CBA(in_ch=cls_out_channels,
                                              out_ch=cls_out_channels,
                                              kernel_size=3,
                                              stride=1),
                                          torch.nn.Conv2d(in_channels=cls_out_channels,
                                                          out_channels=num_class,
                                                          kernel_size=1))

        self.dfl = DFL(reg_max=self.reg_max)

        # 解码
        feat_shapes = [[int(input_shape[0] / 8), int(input_shape[1] / 8)],
                       [int(input_shape[0] / 16), int(input_shape[1] / 16)],
                       [int(input_shape[0] / 32), int(input_shape[1] / 32)]]
        self.anchors, self.strides = make_anchors(feat_shapes=feat_shapes, strides=self.stride, grid_cell_offset=0.5)
        self.anchors = self.anchors.transpose(dim0=0, dim1=1)
        self.strides = self.strides.transpose(dim0=0, dim1=1)

    def forward(self, x):
        # backbone
        feat_1, feat_2, feat_3 = self.backbone.forward(x)
        # neck
        p3, p4, p5 = self.neck((feat_1, feat_2, feat_3))
        # yolov8 head
        reg3 = self.p3_reg(p3)
        cls3 = self.p3_cls(p3)
        h3 = torch.cat(tensors=(reg3, cls3), dim=1)

        reg4 = self.p4_reg(p4)
        cls4 = self.p4_cls(p4)
        h4 = torch.cat(tensors=(reg4, cls4), dim=1)

        reg5 = self.p5_reg(p5)
        cls5 = self.p5_cls(p5)
        h5 = torch.cat(tensors=(reg5, cls5), dim=1)

        h3 = h3.view(h3.size(0), self.num_classes + self.reg_max * 4, -1)
        h4 = h4.view(h4.size(0), self.num_classes + self.reg_max * 4, -1)
        h5 = h5.view(h5.size(0), self.num_classes + self.reg_max * 4, -1)

        head_feat = torch.cat(tensors=(h3, h4, h5), dim=2)
        box_out, cls_out = torch.split(tensor=head_feat, split_size_or_sections=[self.reg_max * 4, self.num_classes],
                                       dim=1)
        box_dfl_out = self.dfl(box_out)

        # 边界框坐标解码
        left_top, right_bottom = torch.split(tensor=box_dfl_out, split_size_or_sections=2, dim=1)
        x1y1 = self.anchors - left_top
        x2y2 = self.anchors + right_bottom
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox_xywh_out = torch.cat(tensors=(c_xy, wh), dim=1) * self.strides
        # 类别
        cls_out = torch.sigmoid(cls_out)

        # 模型输出
        model_out = torch.cat(tensors=(bbox_xywh_out, cls_out), dim=1)
        return model_out

if __name__ == "__main__":
    print(__file__)

    in_tensor = torch.rand(1, 3, 640, 480)  # 均匀分布
    model = YOLOV8Net(model_type="n", input_shape=(640, 480))
    model.eval()
    model.to(device="cpu")
    out_tensor = model(in_tensor)
    print(out_tensor.shape)

    torchinfo.summary(model, input_size=(1, 3, 640, 640), device="cpu")

    # export
    # torch.save(model, "yolov8n_yxl.pth")
    # torch.onnx.export(model, (in_tensor,), "yolov8n_yxl.onnx", input_names=["input"], output_names=["output"])
