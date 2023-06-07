import torch
import torch.nn as nn
import math

def autopad(k, p, d): # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Basic CBR operation -> Conv2d+Batchnormalization+ReLU
    """
    def __init__(self, c1, c2, k, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv): # depthwise convolution
    """
    Inherit Conv
    """
    def __init__(self, c1, c2, k, s=1, d=1, act=True):
        super().__init__(self, c1, c2, k, s=s, g=c1, d=d, act=act)


class Mbnv2_block(nn.Module):
    def __init__(self, c1, c2, stride, expansion, block_id, alpha=0.25):
        super().__init__()
        self.pointwise_filters = make_divisble(int(c2*alpha), 8)
        self.in_channels = c1
        self.block_id = block_id
        self.stride = stride
        self.expand_conv = Conv(c1, c1*expansion, k=1, s=1, p=0, act=True)

        self.dwconv = DWConv(c1*expansion, c1*expansion, k=3, s=self.stride)
        self.pointwise_conv = Conv(c1*expansion, self.pointwise_filters, k=1, act=False)

    def forward(self, x):
        identity = x.copy()
        if self.block_id:
            x = self.expand_conv(x)
        out = self.dwconv(x)
        out = self.pointwise_conv(out)

        if self.stride == 1 and self.pointwise_filters == self.in_channels:
            out = torch.add(out, identity)

        return out


def make_divisble(self, v, divisor, mini_value=None):
    if not mini_value:
        mini_value = divisor

    new_v = max(mini_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



