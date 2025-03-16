import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lib.model.cma import CMA
from lib.model.vector_decoder import VectorFieldDecoder
from lib.model.mask_conv import SparseConvNet
from lib.model.mmd import MultiModalDistillation
from lib.model.pose_est import KeypointPoseEstimation
from lib.model.depth_est import DepthEstimation

from lib.config import cfgs

class MTGOPE(nn.Module):
    def __init__(self):
        super(MTGOPE, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.cma = CMA()
        self.vector_field_decoder = VectorFieldDecoder()
        self.mask_conv = SparseConvNet()

        self.mmd = MultiModalDistillation()

        self.is_training = True

    def pose_head(self, output):
        return KeypointPoseEstimation(output)

    def forward(self, rgb):
        # Backbone
        x = self.backbone.conv1(rgb)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        layer0 = x.clone()
        x = self.backbone.maxpool(x)
        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        features = [layer0, layer1, layer2, layer3, layer4]

        vector_features = self.vector_field_decoder(features)
        depth_features, seg_features = self.cma(features)
        mask = F.softmax(seg_features, dim=1)[:, 0].unsqueeze(1)
        depth_features = self.mask_conv(depth_features, mask)
        outputs = self.mmd(seg_features, depth_features, vector_features)


        segmentation = outputs[:,:cfgs.num_classes,:,:]
        depth = outputs[:,cfgs.num_classes:cfgs.num_classes+2*cfgs.num_depth_bin,:,:]
        vertex = outputs[:,cfgs.num_classes+2*cfgs.num_depth_bin:,:,:]

        b, c_depth, h, w = depth.shape
        ord_num = c_depth // 2
        depth = depth.view(-1, 2, ord_num, h, w)

        outp = {'seg': segmentation, 'vertex': vertex}

        if self.is_training:
            prob = F.log_softmax(depth, dim=1).view(b, c_depth, h, w)
            outp['depth'] = prob

            return outp

        prob = F.softmax(depth, dim=1)[:, 0, :, :, :]
        outp['depth'] = prob

        outp = self.pose_head(outp)

        return outp



if __name__ == '__main__':
    from options import Options

    options = Options()
    opts = options.parse()

    img = torch.randn(1, 3, 480, 640)
    model = MTGOPE()
    model.is_training = False
    out = model(img)
    for i in out:
        print(i)
        print(out[i].shape)
    #
    # print(out['depth_out'])