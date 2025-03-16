import cv2
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils2.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils2.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
from lib.config import cfgs


class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfgs

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]
        print(anno['mask_path'])
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        print(path)
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)

        return inp, kpt_2d, mask

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask = self.read_data(img_id)
        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': torch.tensor(mask.astype(np.uint8)), 'vertex': torch.tensor(vertex), 'img_id': img_id, 'meta': {}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)
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
    dataset = Dataset('/home/ana/Study/Pose/clean-pvnet/data/linemode/cat/train.json', '1', 'train')
    output = dataset.__getitem__((0, 480, 640))
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