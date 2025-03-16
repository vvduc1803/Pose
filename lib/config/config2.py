from yacs.config import CfgNode as CN
import argparse
import os
import open3d

cfgs = CN()

# model
cfgs.model = 'hello'
cfgs.model_dir = 'data/model'
cfgs.det_model = ''
cfgs.kpt_model = ''

# network
cfgs.network = 'dla_34'

# network heads
cfgs.heads = ''

# task
cfgs.task = ''

# gpus
cfgs.gpus = [0, 1, 2, 3]

# if load the pretrained network
cfgs.resume = True

# epoch
cfgs.ep_iter = -1
cfgs.save_ep = 5
cfgs.eval_ep = 5

cfgs.demo_path = 'demo_images/cat'

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfgs.train = CN()

cfgs.train.dataset = 'CocoTrain'
cfgs.train.epoch = 140
cfgs.train.num_workers = 8

# use adam as default
cfgs.train.optim = 'adam'
cfgs.train.lr = 1e-4
cfgs.train.weight_decay = 5e-4

cfgs.train.warmup = False
cfgs.train.milestones = [80, 120]
cfgs.train.gamma = 0.5

cfgs.train.batch_size = 4

#augmentation
cfgs.train.affine_rate = 0.
cfgs.train.cropresize_rate = 0.
cfgs.train.rotate_rate = 0.
cfgs.train.rotate_min = -30
cfgs.train.rotate_max = 30

cfgs.train.overlap_ratio = 0.8
cfgs.train.resize_ratio_min = 0.8
cfgs.train.resize_ratio_max = 1.2

cfgs.train.batch_sampler = ''

# test
cfgs.test = CN()
cfgs.test.dataset = 'CocoVal'
cfgs.test.batch_size = 1
cfgs.test.epoch = -1
cfgs.test.icp = False
cfgs.test.un_pnp = False
cfgs.test.vsd = False
cfgs.test.det_gt = False

cfgs.test.batch_sampler = ''

cfgs.det_meta = CN()
cfgs.det_meta.arch = 'dla'
cfgs.det_meta.num_layers = 34
cfgs.det_meta.heads = CN({'ct_hm': 1, 'wh': 2})

# recorder
cfgs.record_dir = 'data/record'

# result
cfgs.result_dir = 'data/result'

# evaluation
cfgs.skip_eval = False

# dataset
cfgs.cls_type = 'cat'

# tless
cfgs.tless = CN()
cfgs.tless.pvnet_input_scale = (256, 256)
cfgs.tless.scale_train_ratio = (1.8, 2.4)
cfgs.tless.scale_ratio = 2.4
cfgs.tless.box_train_ratio = (1.0, 1.2)
cfgs.tless.box_ratio = 1.2
cfgs.tless.rot = 360.
cfgs.tless.ratio = 0.8

_heads_factory = {
    'pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct_pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct': CN({'ct_hm': 30, 'wh': 2})
}


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    if cfg.task in _heads_factory:
        cfg.heads = _heads_factory[cfg.task]

    if 'Tless' in cfg.test.dataset and cfg.task == 'pvnet':
        cfg.cls_type = '{:02}'.format(int(cfg.cls_type))

    if 'Ycb' in cfg.test.dataset and cfg.task == 'pvnet':
        cfg.cls_type = '{}'.format(int(cfg.cls_type))

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    cfgs.merge_from_file(args.cfg_file)
    opts_idx = [i for i in range(0, len(args.opts), 2) if args.opts[i].split('.')[0] in cfgs.keys()]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    cfgs.merge_from_list(opts)
    parse_cfg(cfgs, args)
    return cfgs


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="/home/ana/Study/Pose/clean-pvnet/configs/linemod.yaml", type=str)
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
if len(args.type) > 0:
    cfgs.task = "run"
cfgs = make_cfg(args)
