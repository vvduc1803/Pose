import os
import json
import numpy as np
from PIL import Image
import tqdm
from config import *
from utils import OpenGLRenderer

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def record_occ_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = 'data/occlusion_linemod'
    # model_meta['data_root'] = data_root
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    inds = np.loadtxt(os.path.join('data/linemod', cls, 'test_occlusion.txt'), np.str)
    inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

    rgb_dir = os.path.join(data_root, 'RGB-D/rgb_noseg')
    for ind in tqdm.tqdm(inds):
        img_name = 'color_{:05d}.png'.format(ind)
        rgb_path = os.path.join(rgb_dir, img_name)
        pose_dir = os.path.join(data_root, 'blender_poses', cls)
        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        if not os.path.exists(pose_path):
            continue

        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose = np.load(pose_path)

        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(data_root, 'masks', cls, '{}.png'.format(ind))
        depth_path = os.path.join(data_root, 'RGB-D', 'depth_noseg',
                'depth_{:05d}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'depth_path': depth_path,
                'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'render', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id

def _linemod_to_coco(objects, split):
    data_root = './data/linemod'

    Ks = []
    corner_3ds = []
    center_3ds = []
    fps_3ds = []

    for cls in objects:
        model_path = os.path.join(data_root, cls, cls+'.ply')

        renderer = OpenGLRenderer(model_path)
        K = linemod_K

        model = renderer.model['pts'] / 1000
        corner_3d = get_model_corners(model)
        center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
        fps_3d = np.loadtxt(os.path.join(data_root, cls, 'farthest.txt'))

        Ks.append(K)
        corner_3ds.append(corner_3d)
        center_3ds.append(center_3d)
        fps_3ds.append(fps_3d)

    models_meta = {
        'K': Ks,
        'corner_3d': corner_3ds,
        'center_3d': center_3ds,
        'fps_3d': fps_3ds,
        'data_root': data_root,
        'cls': objects,
        'split': split
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_occ_ann(models_meta, img_id, ann_id, images, annotations)

    categories = [{'supercategory': 'none', 'id': 1, 'name': cls}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, cls, split + '.json')
    # with open(anno_path, 'w') as f:
    #     json.dump(instance, f)

if __name__ == '__main__':
    cls = ['ape', 'benchvise', 'cam', 'can', 'driller', 'duck', 'eggbox', 'fuse', 'gule', 'holepuncher']
    _linemod_to_coco(cls, 'train')

