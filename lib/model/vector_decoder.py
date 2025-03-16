import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lib.config import cfgs

def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True)

def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')

class VectorFieldDecoder(nn.Module):
    def __init__(self):
        super(VectorFieldDecoder, self).__init__()

        num_units = 512
        num_classes = 9

        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        # self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD, sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)
        self.dropout = nn.Dropout()
        #
        # if cfg.TRAIN.VERTEX_REG:
        #     # center regression branch
        self.conv4_vertex_embed = conv(512, 2 * num_units, kernel_size=1, relu=False)
        self.conv5_vertex_embed = conv(512, 2 * num_units, kernel_size=1, relu=False)
        self.upsample_conv5_vertex_embed = upsample(2.0)
        self.upsample_vertex_embed = upsample(8.0)
        self.conv_vertex_score = conv(2 * num_units, cfgs.in_channels_v, kernel_size=1, relu=False)

    def forward(self, input_features):

        out_conv4_3 = input_features[2]
        out_conv5_vertex_embed = input_features[3]

        # center regression branch
        out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
        # out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
        out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
        out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
        out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
        out_vertex = self.conv_vertex_score(out_vertex_embed_up)

        return out_vertex