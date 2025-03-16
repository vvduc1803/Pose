from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfgs
import pycocotools.coco as coco
import numpy as np
import matplotlib.pyplot as plt
from lib.utils2 import img_utils
import matplotlib.patches as patches
from lib.utils2.pvnet import pvnet_pose_utils


mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfgs.test.dataset)
        self.ann_file = args['ann_file']
        a = '/home/ana/Study/Pose/clean-pvnet/data/linemod/cat/train.json'
        self.coco = coco.COCO(a)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'], mean, std)
        kpt_2d = batch['kpt_2d'].detach().cpu().numpy()

        img_id = 1
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=2))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        corner_2d_gt = corner_2d_gt.astype(np.int32)
        corner_2d_pred = corner_2d_pred.astype(np.int32)

        _, ax = plt.subplots(1)
        ax.imshow(batch['inp'])
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()
        plt.savefig('test.jpg')

    def visualize_demo(self, output, inp, meta):
        inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        kpt_3d = np.array(meta['kpt_3d'])
        K = np.array(meta['K'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(meta['corner_3d'])
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'].detach().cpu().numpy()
        vertex = batch['vertex'][0].detach().cpu().numpy()
        img_id = 1
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)





