from lib.model.depth_decoder import DepthDecoder
from lib.model.multi_embedding import MultiEmbedding
from lib.model.seg_decoder import SegDecoder
from lib.model.utils import *
from lib.config import cfgs

import torch
import torch.nn as nn
import numpy as np


class CMA(nn.Module):
    def __init__(self):
        super(CMA, self).__init__()

        self.scales = cfgs.scales

        self.cma_layers = cfgs.cma_layers

        self.num_ch_dec = cfgs.num_ch_dec
        self.num_ch_enc = cfgs.num_ch_enc
        in_channels_list = cfgs.in_channels_list

        self.depth_decoder = DepthDecoder(self.num_ch_enc, self.num_ch_dec, num_output_channels=cfgs.num_output_channels_depth,
                                          scales=cfgs.scales)
        self.seg_decoder = SegDecoder(self.num_ch_enc, self.num_ch_dec, num_output_channels=cfgs.num_output_channels_seg,
                                      scales=[0])

        att_d_to_s = {}
        att_s_to_d = {}
        for i in self.cma_layers:
            att_d_to_s[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=cfgs.num_head,
                                                ratio=cfgs.head_ratio)
            att_s_to_d[str(i)] = MultiEmbedding(in_channels=in_channels_list[i],
                                                num_head=cfgs.num_head,
                                                ratio=cfgs.head_ratio)
        self.att_d_to_s = nn.ModuleDict(att_d_to_s)
        self.att_s_to_d = nn.ModuleDict(att_s_to_d)

    def forward(self, input_features):

        x = input_features[-1]
        x_d = None
        x_s = None
        depth_outputs = None
        seg_outputs = None
        for i in range(4, -1, -1):
            if x_d is None:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x)
            else:
                x_d = self.depth_decoder.decoder[-2 * i + 8](x_d)

            if x_s is None:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x)
            else:
                x_s = self.seg_decoder.decoder[-2 * i + 8](x_s)

            x_d = [upsample(x_d)]
            x_s = [upsample(x_s)]

            if i > 0:
                x_d += [input_features[i - 1]]
                x_s += [input_features[i - 1]]

            x_d = torch.cat(x_d, 1)
            x_s = torch.cat(x_s, 1)

            x_d = self.depth_decoder.decoder[-2 * i + 9](x_d)
            x_s = self.seg_decoder.decoder[-2 * i + 9](x_s)


            if (i - 1) in self.cma_layers:
                if len(self.cma_layers) == 1:
                    x_d_att = self.att_d_to_s(x_d, x_s)
                    x_s_att = self.att_s_to_d(x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att
                else:
                    x_d_att = self.att_d_to_s[str(i - 1)](x_d, x_s)
                    x_s_att = self.att_s_to_d[str(i - 1)](x_s, x_d)
                    x_d = x_d_att
                    x_s = x_s_att

            if i in self.scales:
                outs = self.depth_decoder.decoder[10 + i](x_d)
                if i == 0:
                    seg_out = self.seg_decoder.decoder[10 + i](x_s)
                    depth_out = outs
                    depth_outputs = depth_out
                    seg_outputs = seg_out
                    # seg_outputs[("seg_logits", i)] = outs[:, :19, :, :]

        return depth_outputs, seg_outputs

if __name__ == '__main__':
    input1 = torch.randn(1, 64, 240, 320)
    input2 = torch.randn(1, 256, 120, 160)
    input3 = torch.randn(1, 512, 60, 80)
    input4 = torch.randn(1, 1024, 30, 40)
    input5 = torch.randn(1, 2048, 15, 20)

    input = [input1, input2, input3, input4, input5]


    model = CMA()
    a, b = model(input)