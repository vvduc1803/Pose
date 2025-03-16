import argparse
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 default='full_res18_192x640',
                                 help="the name of the folder to save the model in",
                                 )

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "test"],
                                 default="eigen_zhou")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--reprojection",
                                 default=1.0,
                                 type=float)
        self.parser.add_argument("--pretrained",
                                 type=int,
                                 default=1,
                                 help='use ImageNet pretrained weight for ResNet encoder')

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size for a single gpu",
                                 default=1)

        self.parser.add_argument("--semantic_distil",
                                 type=float,
                                 default=0.3,
                                 help='weight factor of CE loss for training semantic segmentation')

        self.parser.add_argument("--sgt", type=float, default=0.1, help='weight factor for sgt loss')
        self.parser.add_argument("--sgt_layers", nargs='+', type=int, default=[3, 2, 1],
                                 help='layer configurations for sgt loss')
        self.parser.add_argument("--sgt_margin", type=float, default=0.3, help='margin for sgt loss')
        self.parser.add_argument("--sgt_kernel_size", type=int, nargs='+', default=[5, 5, 5],
                                 help='kernel size (local patch size) for sgt loss')

        self.parser.add_argument("--no_cma", action='store_true', default=False, help='disable cma module')
        self.parser.add_argument("--num_head", type=int, default=4, help='number of embeddings H for cma module')
        self.parser.add_argument("--head_ratio", type=float, default=2, help='embedding dimension ratio for cma module')
        self.parser.add_argument("--cma_layers", nargs="+", type=int, default=[2, 1],
                                 help='layer configurations for cma module')

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
