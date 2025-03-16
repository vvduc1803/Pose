import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.dataset.linemod import config
from lib.dataset.utils import (crop_or_padding_to_fixed_size, rotate_instance,
                               crop_resize_instance_v1, read_linemod_mask, compute_vertex, read_linemod_depth)
import random
import torch

import torchvision.transforms as transforms

from lib.config import cfgs

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class LinemodDataset(data.Dataset):

    def __init__(self, ann_file):
        super(LinemodDataset, self).__init__()

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=mean, std=std)  # Normalize tensor
        ])

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        path_temp = f'{cfgs.root}/{path[32:]}'
        inp = Image.open(path_temp)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = config.linemod_cls_names.index(anno['cls']) + 1
        mask_path_temp = f"{cfgs.root}/{anno['mask_path'][32:]}"
        mask = read_linemod_mask(mask_path_temp, anno['type'], cls_idx)
        temp = path_temp.split('/')[1:-2]
        root = '/'.join(temp)
        depth_file = anno['depth_path'].split('/')[-1]
        depth_path = os.path.join('/', root, 'data', depth_file)
        depth = read_linemod_depth(depth_path)

        depth = depth * mask
        return inp, kpt_2d, mask, depth

    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, index_tuple):
        index = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask, depth = self.read_data(img_id)

        vertex = compute_vertex(mask, kpt_2d).transpose(2, 0, 1)

        img_tensor = self.to_tensor(img)
        ret = {
            'inp': img_tensor,  # Use tensor instead of PIL Image
            'mask': torch.tensor(mask.astype(np.uint8)),
            'vertex': torch.tensor(vertex),
            'depth': torch.tensor(depth.astype(np.int32)),
            'img_id': img_id
        }
        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask

if __name__ == '__main__':
    # dataset = LinemodDataset('/home/ana/Study/Pose/clean-pvnet/data/linemode/cat/train.json')
    dataset = LinemodDataset('/media/multimediateam/4TB/ducfpt/Pose/clean-pvnet/data/linemode/cat/train.json')
    for i in range(len(dataset)):
        # print(i)
        # img, kpt_2d, mask, depth = dataset[i]
        print(i)
        output = dataset.__getitem__(i)
    # for i in output.keys():
    #     print(i, output[i].shape)
    #
    #
    # a = np.array(output['mask'], dtype=np.uint8) * 255
    # print(np.unique(a))
    # cv2.imwrite('a.jpg', a)
    #
    # from lib.visualizers.linemod.pvnet import Visualizer
    # visualizer = Visualizer()
    # visualizer.visualize(1, output)
    # cv2.waitKey(0)

    # with open('/home/ana/Study/Pose/clean-pvnet/data/linemod/cat/data/depth412.dpt', 'rb') as file:
    # with open('/home/ana/Study/Pose/clean-pvnet/data/linemod/cat/data/depth412.dpt', 'r', encoding='latin-1') as file:
    #     content = file.read()
    #     print(content)

    # depth_data = np.fromfile('/home/ana/Study/Pose/clean-pvnet/data/linemod/cat/data/depth1.dpt', dtype=np.float32)
    # with open('/home/ana/Study/Pose/clean-pvnet/data/linemod/cat/data/depth1.dpt') as f:
    #     h, w = np.fromfile(f, dtype=np.uint32, count=2)
    #     data = np.fromfile(f, dtype=np.uint16, count=w * h)
    #     depth = data.reshape((h, w))

    # print(depth.shape)
    # print(np.unique(depth))