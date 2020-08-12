import torch
import torch.nn as nn
from pycls.core import net
from pycls.core.config import cfg

from .weightnet import WeightNet, WeightNet_DW


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert num_channels % 4 == 0
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


def get_block_fun(name):
    """Retrieves the shuffle block function by name."""
    block_funs = {"shufflenet": ShuffleV2Block, "weightnet": WeightNetBlock}
    assert name in block_funs.keys(), "Block function '{}' not supported".format(name)
    return block_funs[name]


class ShuffleStem(nn.Module):
    """Stem of ShuffleNet."""

    def __init__(self, w_in, w_out):
        super(ShuffleStem, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        # 3x3, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_maxpool2d(cx, 3, 2, 1)
        return cx


class ShuffleStage(nn.Module):
    """Stage of ShuffleNet."""

    def __init__(self, w_in, w_out, d, s):
        super(ShuffleStage, self).__init__()
        self._construct(w_in, w_out, d, s)

    def _construct(self, w_in, w_out, d, s):
        block = get_block_fun(cfg.SHUFFLENET.BLOCK_FUN)
        # Construct the blocks
        for i in range(d):
            if i == 0:
                shuffle_block = block(w_in, w_out, w_out // 2, 3, s, True)
            else:
                shuffle_block = block(w_out // 2, w_out, w_out // 2, 3)
            self.add_module("b{}".format(i + 1), shuffle_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, d, s):
        block = get_block_fun(cfg.SHUFFLENET.BLOCK_FUN)
        for i in range(d):
            if i == 0:
                cx = block.complexity(cx, w_in, w_out, w_out // 2, 3, s, True)
            else:
                cx = block.complexity(cx, w_out // 2, w_out, w_out // 2, 3, 1, False)
        return cx


class ShuffleLast(nn.Module):
    """Last of ShuffleNet."""

    def __init__(self, w_in, w_out):
        super(ShuffleLast, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 1, 1, 0, bias=False)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ShuffleHead(nn.Module):
    """ShuffleNet head."""

    def __init__(self, w_in, nc, dropout=False):
        super(ShuffleHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = nn.Identity()
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx


# https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, ksize, stride=1, proj=False):
        super(ShuffleV2Block, self).__init__()
        assert stride in [1, 2]
        self.proj, pad, outputs = proj, ksize // 2, oup - inp

        self.branch_main_fc_conv = nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False)
        self.branch_main_fc_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        self.branch_main_dw_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            ksize,
            stride,
            pad,
            groups=mid_channels,
            bias=False,
        )
        self.branch_main_dw_bn = nn.BatchNorm2d(
            mid_channels, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )

        self.branch_main_pw_conv = nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False)
        self.branch_main_pw_bn = nn.BatchNorm2d(
            outputs, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.branch_main_pw_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        if self.proj:
            self.branch_proj_dw_conv = nn.Conv2d(
                inp, inp, ksize, stride, pad, groups=inp, bias=False
            )
            self.branch_proj_dw_bn = nn.BatchNorm2d(
                inp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )

            self.branch_proj_pw_conv = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
            self.branch_proj_pw_bn = nn.BatchNorm2d(
                inp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )
            self.branch_proj_pw_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, old_x):
        if self.proj:
            x_proj = self.branch_proj_dw_conv(old_x)
            x_proj = self.branch_proj_dw_bn(x_proj)
            x_proj = self.branch_proj_pw_conv(x_proj)
            x_proj = self.branch_proj_pw_bn(x_proj)
            x_proj = self.branch_proj_pw_relu(x_proj)

            x = self.branch_main_fc_conv(old_x)
            x = self.branch_main_fc_relu(x)
            x = self.branch_main_dw_conv(x)
            x = self.branch_main_dw_bn(x)
            x = self.branch_main_pw_conv(x)
            x = self.branch_main_pw_bn(x)
            x = self.branch_main_pw_relu(x)
        else:
            x_proj, x = channel_shuffle(old_x)
            x = self.branch_main_fc_conv(x)
            x = self.branch_main_fc_relu(x)
            x = self.branch_main_dw_conv(x)
            x = self.branch_main_dw_bn(x)
            x = self.branch_main_pw_conv(x)
            x = self.branch_main_pw_bn(x)
            x = self.branch_main_pw_relu(x)
        return torch.cat((x_proj, x), 1)

    @staticmethod
    def complexity(cx, w_in, w_out, w_mid, ksize, stride, proj):
        if proj:
            cx = net.complexity_conv2d(
                cx, w_in, w_in, ksize, stride, ksize // 2, groups=w_in, bias=False
            )
            cx = net.complexity_batchnorm2d(cx, w_in)
            cx = net.complexity_conv2d(cx, w_in, w_in, 1, 1, 0, bias=False)
            cx = net.complexity_batchnorm2d(cx, w_in)

            cx = net.complexity_conv2d(cx, w_in, w_mid, 1, 1, 0, bias=False)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = net.complexity_conv2d(
                cx, w_mid, w_mid, ksize, 1, ksize // 2, groups=w_mid, bias=False
            )
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = net.complexity_conv2d(cx, w_mid, w_out, 1, 1, 0, bias=False)
            cx = net.complexity_batchnorm2d(cx, w_out)
        else:
            # TODO: add the complexity of channel_shuffle
            cx = net.complexity_conv2d(cx, w_in, w_mid, 1, 1, 0, bias=False)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = net.complexity_conv2d(
                cx, w_mid, w_mid, ksize, stride, ksize // 2, groups=w_mid, bias=False
            )
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = net.complexity_conv2d(cx, w_mid, w_out, 1, 1, 0, bias=False)
            cx = net.complexity_batchnorm2d(cx, w_out)
        return cx

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class WeightNetBlock(nn.Module):
    def __init__(self, inp, oup, mid_channels, ksize, stride=1, proj=False):
        super(WeightNetBlock, self).__init__()
        assert stride in [1, 2]
        self.proj, outputs = proj, oup - inp

        self.reduce = nn.Conv2d(inp, max(16, inp // 16), 1, 1, 0, bias=True)

        self.branch_main_fc_conv = WeightNet(inp, mid_channels, 1, 1)
        self.branch_main_fc_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        self.branch_main_dw_conv = WeightNet_DW(mid_channels, ksize, stride)
        self.branch_main_dw_bn = nn.BatchNorm2d(
            mid_channels, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )

        self.branch_main_pw_conv = WeightNet(mid_channels, outputs, 1, 1)
        self.branch_main_pw_bn = nn.BatchNorm2d(
            outputs, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.branch_main_pw_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        if self.proj:
            self.branch_proj_dw_conv = WeightNet_DW(inp, ksize, stride)
            self.branch_proj_dw_bn = nn.BatchNorm2d(
                inp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )

            self.branch_proj_pw_conv = WeightNet(inp, inp, 1, 1)
            self.branch_proj_pw_bn = nn.BatchNorm2d(
                inp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
            )
            self.branch_proj_pw_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, old_x):
        if self.proj:
            x_proj, x = old_x, old_x
        else:
            x_proj, x = channel_shuffle(old_x)
        x_gap = self.reduce(x.mean(dim=[2, 3], keepdim=True))

        x = self.branch_main_fc_conv(x, x_gap)
        x = self.branch_main_fc_relu(x)
        x = self.branch_main_dw_conv(x, x_gap)
        x = self.branch_main_dw_bn(x)
        x = self.branch_main_pw_conv(x, x_gap)
        x = self.branch_main_pw_bn(x)
        x = self.branch_main_pw_relu(x)
        if self.proj:
            x_proj = self.branch_proj_dw_conv(x_proj, x_gap)
            x_proj = self.branch_proj_dw_bn(x_proj)
            x_proj = self.branch_proj_pw_conv(x_proj, x_gap)
            x_proj = self.branch_proj_pw_bn(x_proj)
            x_proj = self.branch_proj_pw_relu(x_proj)

        return torch.cat((x_proj, x), 1)

    @staticmethod
    def complexity(cx, w_in, w_out, w_mid, ksize, stride, proj):
        cx_h, cx_w = cx["h"], cx["w"]
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, max(16, w_in // 16), 1, 1, 0, bias=True)
        cx["h"], cx["w"] = cx_h, cx_w
        if proj:
            cx = WeightNet_DW.complexity(cx, w_in, ksize, stride)
            cx = net.complexity_batchnorm2d(cx, w_in)
            cx = WeightNet.complexity(cx, w_in, w_in, 1, 1)
            cx = net.complexity_batchnorm2d(cx, w_in)

            cx = WeightNet.complexity(cx, w_in, w_mid, 1, 1)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = WeightNet_DW.complexity(cx, w_mid, ksize, stride)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = WeightNet.complexity(cx, w_mid, w_out, 1, 1)
            cx = net.complexity_batchnorm2d(cx, w_out)
        else:
            # TODO: add the complexity of channel_shuffle
            cx = WeightNet.complexity(cx, w_in, w_mid, 1, 1)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = WeightNet_DW.complexity(cx, w_mid, ksize, stride)
            cx = net.complexity_batchnorm2d(cx, w_mid)
            cx = WeightNet.complexity(cx, w_mid, w_out, 1, 1)
            cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ShuffleNetV2(nn.Module):
    def __init__(self):
        super(ShuffleNetV2, self).__init__()
        assert cfg.TRAIN.DATASET in [
            "imagenet"
        ], "Training ShuffleNet on {} is not supported".format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in [
            "imagenet"
        ], "Testing ShuffleNet on {} is not supported".format(cfg.TEST.DATASET)
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats, model_size = [4, 8, 4], cfg.SHUFFLENET.MODEL_SIZE
        if model_size == "0.5x":
            self.stage_out_channels = [3, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self.stage_out_channels = [3, 24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self.stage_out_channels = [3, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self.stage_out_channels = [3, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        self.stem = ShuffleStem(self.stage_out_channels[0], self.stage_out_channels[1])
        self.s1 = ShuffleStage(
            self.stage_out_channels[1],
            self.stage_out_channels[2],
            self.stage_repeats[0],
            2,
        )
        self.s2 = ShuffleStage(
            self.stage_out_channels[2],
            self.stage_out_channels[3],
            self.stage_repeats[1],
            2,
        )
        self.s3 = ShuffleStage(
            self.stage_out_channels[3],
            self.stage_out_channels[4],
            self.stage_repeats[2],
            2,
        )
        self.s4 = ShuffleLast(self.stage_out_channels[4], self.stage_out_channels[5])
        self.head = ShuffleHead(
            self.stage_out_channels[5], cfg.MODEL.NUM_CLASSES, model_size == "2.0x"
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "stem" in n:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        stage_repeats, model_size = [4, 8, 4], cfg.SHUFFLENET.MODEL_SIZE
        if model_size == "0.5x":
            stage_out_channels = [3, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            stage_out_channels = [3, 24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            stage_out_channels = [3, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            stage_out_channels = [3, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        cx = ShuffleStem.complexity(cx, stage_out_channels[0], stage_out_channels[1])
        cx = ShuffleStage.complexity(
            cx, stage_out_channels[1], stage_out_channels[2], stage_repeats[0], 2
        )
        cx = ShuffleStage.complexity(
            cx, stage_out_channels[2], stage_out_channels[3], stage_repeats[1], 2
        )
        cx = ShuffleStage.complexity(
            cx, stage_out_channels[3], stage_out_channels[4], stage_repeats[2], 2
        )
        cx = ShuffleLast.complexity(cx, stage_out_channels[4], stage_out_channels[5])
        cx = ShuffleHead.complexity(cx, stage_out_channels[5], cfg.MODEL.NUM_CLASSES)
        return cx
