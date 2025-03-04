import torch
import math
import torch.nn as nn
from model.common import *
import torch.nn.functional as F

class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_up = False,use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        self.head = conv(n_colors, n_feats, kernel_size)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)
        self.use_up = use_up
        if use_up:
            self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        if self.use_up:
            y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)
        

        return y


class SSPSR(nn.Module):
    def __init__(self, n_subs = 8, n_ovls = 2, n_colors=128, n_blocks=3, n_feats=256, n_scale = 4,res_scale=.1, use_share=True, conv=default_conv):
        super(SSPSR, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)
        self.n_scale = n_scale
        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        if self.shared:
            self.branch = BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, use_up = False,up_scale=n_scale, conv=default_conv)
            # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
        else:
            self.branch = nn.ModuleList()
            for i in range(self.G):
                self.branch.append(BranchUnit(n_subs, n_feats, n_blocks, act, res_scale, use_up = False,up_scale=n_scale, conv=default_conv))

        self.trunk = BranchUnit(n_colors, n_feats, n_blocks, act, res_scale,use_up = True, up_scale=n_scale, use_tail=False, conv=default_conv)
        self.skip_conv = conv(n_colors, n_feats, kernel_size)
        self.final = conv(n_feats, n_colors, kernel_size)
        

    def forward(self, x):
        b, c, h, w = x.shape
        lms = F.interpolate(x,scale_factor=self.n_scale,mode='bicubic')
        # Initialize intermediate “result”, which is upsampled with n_scale//2 times
        y = torch.zeros(b, c, h,  w).to(x.device) 

        channel_counter = torch.zeros(c).to(x.device) 

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self.branch[g](xi)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        
        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y
