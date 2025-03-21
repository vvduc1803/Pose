from .dla import get_network as get_dla
from lib.utils2.tless import tless_config


_network_factory = {
    'dla': get_dla
}


def get_network(cfg):
    arch = cfg.network
    heads = cfg.heads
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    network = get_model(num_layers, heads, head_conv, tless_config.down_ratio, cfg.det_dir)
    return network

