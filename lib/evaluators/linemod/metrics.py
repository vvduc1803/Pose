import torch
import math
import scipy
import numpy as np

from lib.config import cfgs
import pycocotools.coco as coco
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
from lib.utils.depth_utils import create_sparse_depth
import os
from lib.utils.linemod import linemod_config
import torch
from lib.utils.vsd import inout


class Evaluator:
    def __init__(self):

        self.ann_file = cfgs.ann_file
        self.coco = coco.COCO(self.ann_file)

        cls = cfgs.cls_type
        model_path = os.path.join(cfgs.root, cfgs.id, cls, cls + '.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = linemod_config.diameters[cls] / 100

        self.proj2d = []
        self.add = []
        self.cmd5 = []

        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []

        self.mask_ap = []

        self.abs_rel = []
        self.sq_rel = []
        self.rmse = []
        self.rmse_log = []
        self.d1 = []
        self.d2 = []
        self.d3 = []

        self.height = 480
        self.width = 640

        model = inout.load_ply(model_path)
        model['pts'] = model['pts'] * 1000

    def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def depth_metrics(self, gt_spare, pred_dense):
        """Computation of error metrics between predicted and ground truth depths
        """
        pred_spare = create_sparse_depth(pred_dense)
        valid_mask = gt_spare > 0
        pred_spare = pred_spare[valid_mask]
        gt_spare = gt_spare[valid_mask]

        thresh = torch.max((gt_spare / pred_spare), (pred_spare / gt_spare))
        d1 = float((thresh < 1.25).float().mean())
        d2 = float((thresh < 1.25 ** 2).float().mean())
        d3 = float((thresh < 1.25 ** 3).float().mean())

        rmse = (gt_spare - pred_spare) ** 2
        rmse = math.sqrt(rmse.mean())

        rmse_log = (torch.log(gt_spare) - torch.log(pred_spare)) ** 2
        rmse_log = math.sqrt(rmse_log.mean())

        abs_rel = ((gt_spare - pred_spare).abs() / gt_spare).mean()
        sq_rel = (((gt_spare - pred_spare) ** 2) / gt_spare).mean()

        self.abs_rel.append(abs_rel)
        self.sq_rel.append(sq_rel)
        self.rmse.append(rmse)
        self.rmse_log.append(rmse_log)
        self.d1.append(d1)
        self.d2.append(d2)
        self.d3.append(d3)

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        # if cfgs.cls_type in ['eggbox', 'glue']:
        #     self.add_metric(pose_pred, pose_gt, syn=True)
        # else:
        self.add_metric(pose_pred, pose_gt)
        self.projection_2d(pose_pred, pose_gt, K)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)
        self.depth_metrics(batch['depth'], output['depth'])



    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)

        abs_rel = np.mean(self.abs_rel)
        sq_rel = np.mean(self.sq_rel)
        rmse = np.mean(self.rmse)
        rmse_log = np.mean(self.rmse_log)
        d1 = np.mean(self.d1)
        d2 = np.mean(self.d2)
        d3 = np.mean(self.d3)
        # print('2d projections metric: {}'.format(proj2d))
        # print('ADD metric: {}'.format(add))
        # print('5 cm 5 degree metric: {}'.format(cmd5))
        # print('mask ap70: {}'.format(ap))

        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []
        self.abs_rel = []
        self.sq_rel = []
        self.rmse = []
        self.rmse_log = []
        self.d1 = []
        self.d2 = []
        self.d3 = []
        return {{'pose': {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap}},
                {'depth': {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'd1': d1, 'd2': d2, 'd3': d3}}}



