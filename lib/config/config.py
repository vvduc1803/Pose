class Config:
    def __init__(self):
        self.cls_type = 'cat'
        self.root = '/home/ana/Study/Pose'

        self.root_old = '/home/ana/Study/Pose/clean-pvnet'

        self.id = 'linemod'
        self.split = 'train'
        self.data_root =  '{}/data/linemod/{}/JPEGImages'.format(self.root, self.cls_type)
        self.ann_file=  '{}/data/linemod/{}/{}.json'.format(self.root, self.cls_type, self.split)

        self.SID = True

        # Linemod information
        self.num_classes = 2
        self.min_depth = 1.
        self.max_depth = 1500.
        self.num_keypoints = 9
        self.num_depth_bin = 40

        # Hyperparameters for CMA model
        self.num_ch_enc = [64, 256, 512, 1024, 2048]
        self.in_channels_list = [32, 64, 128, 256, 16]
        self.scales = range(4)
        self.num_head = 4
        self.head_ratio = 2
        self.cma_layers = [2, 1]

        # Hyperparameters for DepthDecoder
        self.num_output_channels_depth = 80
        self.use_skips = True

        # Hyperparameters for SegDecoder
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.num_output_channels_seg = 2

        # Hyperparameters for MultiModalDistillation
        self.in_channels_s = self.num_classes
        self.in_channels_d = 2 * self.num_depth_bin
        self.in_channels_v = 2 * self.num_keypoints
        self.reduction = 16
        self.spatial_kernel_size = 7

        # Hyperparameters for MTGOPE
        self.pretrained_backbone = False

        # Hyperparameters for VectorFieldDecoder
        self.num_units = 512

        # Hyperparameters for SparseConvNet
        self.sparse_conv_in_channel = 80
        self.sparse_conv_out_channel = 80

        # General hyperparameters
        self.learning_rate = 0.001
        self.batch_size = 1
        self.epochs = 2

        # Losses hyperparameters
        self.seg_classes_weight = [1.0, 100.0]
        self.losses_weight = [10, 1, 1]

# Usage
cfgs = Config()
