import torch
import torch.nn as nn

from common import *


class HumanPose_MBN(nn.Module):
    def __init__(self, c1, c2, alpha=0.5):
        super().__init__()

        first_block_filters = make_divisble(int(32*alpha), 8)
        self.conv0 = Conv(c1, first_block_filters, k=3, s=2)
        self.block0 = Mbnv2_block(first_block_filters, c2=16, stride=1, expansion=1, block_id=0)
        self.block1 = Mbnv2_block(make_divisble(int(16*alpha), 8), c2=16, stride=2, expansion=6, block_id=1)
        self.block2 = Mbnv2_block(make_divisble(int(16*alpha), 8), c2=16, stride=1, expansion=6, block_id=2)