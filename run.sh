#!/bin/sh
python train.py --dataset ax_crop_dataset --dataset2 fa_dataset --arch MultiViewNestedUNet --num_classes=2
