from collections import OrderedDict

def rename_checkpoint_keys(src_cp, dst_cp):
    new_checkpoint = OrderedDict()
    for src, dst in zip(src_cp.state_dict().items(), dst_cp.state_dict().items()):
        new_checkpoint[dst[0]] = src[1]
    return new_checkpoint