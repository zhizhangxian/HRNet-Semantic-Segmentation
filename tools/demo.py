import argparse
import _init_paths

import torch
import models

from config import config
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)       
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()

# print(config.MODEL.PRETRAINED)
# print(config.TRAIN.EXTRA_EPOCH)
# exit()

config.defrost()
config.MODEL.PRETRAINED = False
model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)
if config.TRAIN.OPTIMIZER == 'sgd':

    params_dict = dict(model.named_parameters())
    if config.TRAIN.NONBACKBONE_KEYWORDS:
        bb_lr = []
        nbb_lr = []
        nbb_keys = set()
        for k, param in params_dict.items():
            if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                nbb_lr.append(param)
                nbb_keys.add(k)
            else:
                bb_lr.append(param)
        print(nbb_keys)
        params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
    else:
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

    optimizer = torch.optim.SGD(params,
                            lr=config.TRAIN.LR,
                            momentum=config.TRAIN.MOMENTUM,
                            weight_decay=config.TRAIN.WD,
                            nesterov=config.TRAIN.NESTEROV,
                            )
else:
    raise ValueError('Only Support SGD optimizer')
