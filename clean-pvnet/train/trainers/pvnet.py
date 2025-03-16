import torch.nn as nn
from lib.utils2 import net_utils
import torch


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()

    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()
        vote_loss = self.vote_crit(output['vote_dim'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg_dim'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

if __name__ == '__main__':
    from lib.config import cfgs
    from lib.networks.resnet import get_pose_net

    arch = cfgs.network
    heads = cfgs.heads
    head_conv = cfgs.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    network = get_pose_net(50, heads, head_conv)

    img = torch.randn(1, 3, 480, 640)
    mask = torch.randn(1, 1, 480, 640)
    vertex = torch.randn(1, 18, 480, 640)
    inputs = {'inp': img, 'mask': mask, 'vertex': vertex, 'meta': {}}
    trainer = NetworkWrapper(network)
    output, loss, scalar_stats, image_stats = trainer(inputs)
    for i in loss:
        print(i)
