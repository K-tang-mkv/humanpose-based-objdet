import torch
import torch.nn as nn
from torchsummary import summary

from common import *


class HumanPose_MBN(nn.Module):
    def __init__(self, c1=3, c2=14, alpha=0.5):
        super().__init__()

        first_block_filters = make_divisble(int(32*alpha), 8)
        self.conv0 = Conv(c1, first_block_filters, k=3, s=2)
        self.block0 = Mbnv2_block(first_block_filters, c2=16, stride=1, expansion=1, block_id=0)
        self.block1 = Mbnv2_block(make_divisble(int(16*alpha), 8), c2=16, stride=2, expansion=6, block_id=1)
        self.block2 = Mbnv2_block(make_divisble(int(16*alpha), 8), c2=16, stride=1, expansion=6, block_id=2)
        self.block3 = Mbnv2_block(make_divisble(int(16*alpha), 8), c2=32, stride=2, expansion=6, block_id=3)
        self.block4 = Mbnv2_block(make_divisble(int(32*alpha), 8), c2=32, stride=1, expansion=6, block_id=4)
        self.block5 = Mbnv2_block(make_divisble(int(32*alpha), 8), c2=32, stride=1, expansion=6, block_id=5)
        self.block6 = Mbnv2_block(make_divisble(int(32*alpha), 8), c2=48, stride=2, expansion=6, block_id=6)
        self.block7 = Mbnv2_block(make_divisble(int(48*alpha), 8), c2=48, stride=1, expansion=6, block_id=7)
        self.block8 = Mbnv2_block(make_divisble(int(48*alpha), 8), c2=48, stride=1, expansion=6, block_id=8)
        self.block9 = Mbnv2_block(make_divisble(int(48 * alpha), 8), c2=48, stride=1, expansion=6, block_id=9)
        self.block10 = Mbnv2_block(make_divisble(int(48 * alpha), 8), c2=64, stride=1, expansion=6, block_id=10)
        self.block11 = Mbnv2_block(make_divisble(int(64 * alpha), 8), c2=64, stride=1, expansion=6, block_id=11)
        self.block12 = Mbnv2_block(make_divisble(int(64 * alpha), 8), c2=64, stride=1, expansion=6, block_id=12)

        self.bb_last_conv = Conv(make_divisble(int(64*alpha), 8), c2=1280, k=1)

        # backbone
        self.backbone_sequential = nn.Sequential(
            self.conv0,
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
            self.block12,
            self.bb_last_conv
        )
        self.fm_p2 = self.backbone_sequential[:4]  # feature map with 4 times downsample, output of block2
        self.fm_p3 = self.backbone_sequential[:7]  # p3/8
        self.fm_p4 = self.backbone_sequential[:11]  # p4/16

        # head
        # fuse-1 -------------------------------
        self.fuse1_sequential = nn.Sequential(
            Conv(1280, 64, k=1, act="relu"),
            DWConvTranspose2d(64, 64, k=3, p1=1)
        )
        self.fuse1_conv = Conv(24, 64, k=1, act="relu")

        # fuse-2 -------------------------------
        self.fuse2_sequential = nn.Sequential(
            DWConv(64, 64, k=3, act=False),
            Conv(64, 32, k=1, act="relu"),
            DWConvTranspose2d(32, 32, k=3, s=2, p1=1, p2=1) # output size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        )
        self.fuse2_conv = Conv(16, 32, k=1, act="relu")

        # fuse-3 -------------------------------
        self.fuse3_sequential = nn.Sequential(
            DWConv(32, 32, k=3, act=False),
            Conv(32, 24, k=1, act="relu"),
            DWConvTranspose2d(24, 24, k=3, s=2, p1=1, p2=1)
        )
        self.fuse3_conv = Conv(8, 24, k=1, act="relu")

        # final sequential ---------------------
        self.final_sequential = nn.Sequential(
            DWConv(24, 24, k=3, act=False),
            Conv(24, 24, k=1, act="relu"),
            DWConv(24, 24, k=3, act=False),
            Conv(24, 24, k=1, act="relu"),
            Conv(24, 14, k=1, act=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # extract different scale feature maps to fuse
        fm_p2 = self.fm_p2(x) # feature map with 4 times downsample, output of block2
        fm_p3 = self.fm_p3(x) # p3/8
        fm_p4 = self.fm_p4(x) # p4/16

        backbone_output = self.backbone_sequential[11:](fm_p4)
        print(self.fuse1_sequential(backbone_output).shape, self.fuse1_conv(fm_p4).shape)
        # fuse backbone feature maps
        fuse1 = torch.add(self.fuse1_sequential(backbone_output), self.fuse1_conv(fm_p4))
        fuse2 = torch.add(self.fuse2_sequential(fuse1), self.fuse2_conv(fm_p3))
        fuse3 = torch.add(self.fuse3_sequential(fuse2), self.fuse3_conv(fm_p2))

        # final detect
        output = self.final_sequential(fuse3)

        return output


if __name__ == "__main__":
    from thop import profile
    x = torch.randn((1, 3, 192, 192))

    model = HumanPose_MBN(3, 14).eval()

    summary(model, (3,192,192))
    flops, params = profile(model, (x,))
    print(f"GFLOPs: {flops/1e9}, Params: {params}")