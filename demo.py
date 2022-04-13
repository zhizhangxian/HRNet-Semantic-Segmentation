import sys
import argparse
sys.path.insert(0, 'lib')

from lib.datasets.cityscapes import Cityscapes
from config import config, update_config

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        default='experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                        help='experiment configure file name',
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
crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

ds = Cityscapes(root='/Users/zzx/Datasets/Cityscapes/cityscapes/',
                list_path=config.DATASET.TRAIN_SET,
                num_samples=None,
                num_classes=config.DATASET.NUM_CLASSES,
                multi_scale=config.TRAIN.MULTI_SCALE,
                flip=config.TRAIN.FLIP,
                ignore_label=config.TRAIN.IGNORE_LABEL,
                base_size=config.TRAIN.BASE_SIZE,
                crop_size=crop_size,
                downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                scale_factor=config.TRAIN.SCALE_FACTOR)

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=2,
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    sampler=None)


if __name__ == '__main__':
        
    dl = iter(dl)
    sample = next(dl)
    print(sample)