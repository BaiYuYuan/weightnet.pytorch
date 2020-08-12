import pycls.core.net as net
import torch
import torch.nn as nn
from torch.nn import functional as F


def conv2d_sample_by_sample(
    x: torch.Tensor,
    weight: torch.Tensor,
    oup: int,
    inp: int,
    ksize: int,
    stride: int,
    groups: int,
) -> torch.Tensor:
    padding, batch_size = ksize // 2, x.shape[0]
    if batch_size == 1:
        out = F.conv2d(
            x,
            weight=weight.view(oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups,
        )
    else:
        out = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            weight.view(batch_size * oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups * batch_size,
        )
        out = out.permute([1, 0, 2, 3]).view(
            batch_size, oup, out.shape[2], out.shape[3]
        )
    return out


# https://github.com/megvii-model/WeightNet/blob/master/weightnet.py


class WeightNet(nn.Module):
    r"""Applies WeightNet to a standard convolution.

    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.

    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = nn.Conv2d(inp_gap, M * oup, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(
            M * oup, oup * inp * ksize * ksize, 1, 1, 0, groups=G * oup, bias=False
        )

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.oup, self.inp, self.ksize, self.stride, 1
        )

    @staticmethod
    def complexity(cx, w_in, w_out, ksize, stride, M=2, G=2):
        cx = net.complexity_conv2d(
            cx, max(16, w_in // 16), M * w_out, 1, 1, 0, groups=1, bias=True
        )
        cx = net.complexity_conv2d(
            cx,
            M * w_out,
            w_in * w_out * ksize * ksize,
            1,
            1,
            0,
            groups=G * w_out,
            bias=False,
        )
        cx = net.complexity_conv2d(
            cx, w_in, w_out, ksize, stride, ksize // 2, groups=1, bias=False
        )
        return cx


class WeightNet_DW(nn.Module):
    r""" Here we show a grouping manner when we apply WeightNet to a depthwise convolution.

    The grouped fc layer directly generates the convolutional kernel, has fewer parameters while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.

    """

    def __init__(self, inp, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = nn.Conv2d(inp_gap, M // G * inp, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(
            M // G * inp, inp * ksize * ksize, 1, 1, 0, groups=inp, bias=False
        )

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.inp, 1, self.ksize, self.stride, self.inp
        )

    @staticmethod
    def complexity(cx, w_in, ksize, stride, M=2, G=2):
        cx = net.complexity_conv2d(
            cx, max(16, w_in // 16), M // G * w_in, 1, 1, 0, groups=1, bias=True
        )
        cx = net.complexity_conv2d(
            cx, M // G * w_in, w_in * ksize * ksize, 1, 1, 0, groups=w_in, bias=False
        )
        cx = net.complexity_conv2d(
            cx, w_in, w_in, ksize, stride, ksize // 2, groups=w_in, bias=False
        )
        return cx
