# -*- coding:utf-8 -*-
# @FileName  :cspdarknet.py
# @Time      :2025/10/8 10:08

import torch

def auto_pad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class CBA(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 dilation=1,
                 act=True):
        super().__init__()
        self.padding = auto_pad(kernel_size, padding, dilation)
        self.conv = torch.nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=self.padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=True)
        self.bn = torch.nn.BatchNorm2d(num_features=out_ch,
                                       eps=0.001,
                                       momentum=0.03,
                                       affine=True,
                                       track_running_stats=True)
        self.act = torch.nn.SiLU() if act else torch.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarknetBottleneck(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size_list=(3, 3),
                 groups=1,
                 e=0.5,
                 shortcut=True):
        super().__init__()
        self.mid_ch = int(out_ch * e)
        self.conv_1 = CBA(in_ch=in_ch,
                          out_ch=self.mid_ch,
                          kernel_size=kernel_size_list[0],
                          stride=1)
        self.conv_2 = CBA(in_ch=self.mid_ch,
                          out_ch=out_ch,
                          kernel_size=kernel_size_list[1],
                          stride=1,
                          groups=groups)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)

        return x + x2 if self.add else x2


class C2f(torch.nn.Module):
    def __init__(self, in_ch, out_ch, group=1, n=1, e=0.5, shortcut=False):
        super().__init__()
        self.half_ch = int(out_ch * e)
        self.conv_1 = CBA(in_ch=in_ch,
                          out_ch=2 * self.half_ch,
                          kernel_size=1,
                          stride=1)
        self.conv_2 = CBA(in_ch=(2 + n) * self.half_ch,
                          out_ch=out_ch,
                          kernel_size=1,
                          stride=1)

        self.bottleneck_list = torch.nn.ModuleList(DarknetBottleneck(in_ch=self.half_ch,
                                                                      out_ch=self.half_ch,
                                                                      kernel_size_list=(3, 3),
                                                                      groups=group, e=1.0,
                                                                      shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        x = self.conv_1(x)
        x1, x2 = torch.split(x, self.half_ch, dim=1)
        features = [x1, x2]
        for bottleneck in self.bottleneck_list:
            x2 = bottleneck(x2)
            features.append(x2)

        x = torch.cat(tensors=features, dim=1)
        x_out = self.conv_2(x)
        return x_out


class SPPF(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        mid_ch = in_ch // 2
        self.conv_1 = CBA(in_ch=in_ch,
                          out_ch=mid_ch,
                          kernel_size=1,
                          stride=1)
        self.conv_2 = CBA(in_ch=mid_ch * 4,
                          out_ch=out_ch,
                          kernel_size=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv_1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        x = torch.cat(tensors=[x, y1, y2, y3], dim=1)
        x_out = self.conv_2(x)
        return x_out


# yolo v8
class CSPDarkNetV8(torch.nn.Module):
    def __init__(self, net_depth=1, net_width=16, width_ratio=1.0):
        super().__init__()

        # stage 1
        self.stem = CBA(in_ch=3,
                        out_ch=net_width,
                        kernel_size=3,
                        stride=2)
        # stage 2
        self.dark_2 = torch.nn.Sequential(CBA(in_ch=net_width,
                                              out_ch=net_width * 2,
                                              kernel_size=3,
                                              stride=2),
                                          C2f(in_ch=net_width * 2,
                                              out_ch=net_width * 2,
                                              n=net_depth,
                                              shortcut=True))

        # stage 3
        self.dark_3 = torch.nn.Sequential(CBA(in_ch=net_width * 2,
                                              out_ch=net_width * 4,
                                              kernel_size=3,
                                              stride=2),
                                          C2f(in_ch=net_width * 4,
                                              out_ch=net_width * 4,
                                              n=net_depth * 2,
                                              shortcut=True))

        # stage 4
        self.dark_4 = torch.nn.Sequential(CBA(in_ch=net_width * 4,
                                              out_ch=net_width * 8,
                                              kernel_size=3,
                                              stride=2),
                                          C2f(in_ch=net_width * 8,
                                              out_ch=net_width * 8,
                                              n=net_depth * 2,
                                              shortcut=True))

        # stage 5
        self.dark_5 = torch.nn.Sequential(CBA(in_ch=net_width * 8,
                                              out_ch=int(net_width * 16 * width_ratio),
                                              kernel_size=3,
                                              stride=2),
                                          C2f(in_ch=int(net_width * 16 * width_ratio),
                                              out_ch=int(net_width * 16 * width_ratio),
                                              n=net_depth,
                                              shortcut=True),
                                          SPPF(in_ch=int(net_width * 16 * width_ratio),
                                               out_ch=int(net_width * 16 * width_ratio),
                                               kernel_size=5))

    def forward(self, x):
        x = self.stem(x)  # [1, 16, 320, 320]
        x = self.dark_2(x)

        x = self.dark_3(x)
        feat_1 = x

        x = self.dark_4(x)
        feat_2 = x

        x = self.dark_5(x)
        feat_3 = x

        return feat_1, feat_2, feat_3


if __name__ == "__main__":
    print(__file__)

    in_tensor = torch.randn(1, 3, 640, 640)
    model = CSPDarkNetV8()
    model.eval()
    # out_tensors = model(in_tensor)
    # for out_tensor in out_tensors:
    #     print(out_tensor.shape)
    onnx_path = "backbone.onnx"
    torch.onnx.export(model=model,  # 要导出的模型
                      args=(in_tensor,),  # 示例输入（用于跟踪计算图）
                      f=onnx_path,  # 输出文件路径
                      export_params=True,  # 是否导出模型参数（权重）
                      opset_version=11,  # ONNX 算子集版本（根据部署需求选择）
                      do_constant_folding=True,  # 是否折叠常量（优化推理）
                      input_names=["input"],  # 输入节点的名称（可选，便于后续部署）
                      output_names=["output"])  # 输出节点的名称（可选）
