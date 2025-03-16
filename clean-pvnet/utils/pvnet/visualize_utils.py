import matplotlib.pyplot as plt
from lib.utils2 import img_utils
from lib.utils2.pvnet import pvnet_config

mean = pvnet_config.mean
std = pvnet_config.std


def visualize_ann(img, kpt_2d, mask, savefig=False):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    ax2.imshow(mask)
    if savefig:
        plt.savefig('test.jpg')
    else:
        plt.show()


def visualize_linemod_ann(img, kpt_2d, mask, savefig=False):
    img = img_utils.unnormalize_img(img, mean, std, False)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    ax2.imshow(mask)
    plt.show()

