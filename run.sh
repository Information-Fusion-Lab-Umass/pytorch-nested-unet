#!/bin/sh
python train.py --dataset ax_crop_dataset --dataset2 fa_dataset --arch MultiViewNestedUNet --img_ext .png --mask_ext .png --num_classes=2 --input_w=96 --input_h=96 --batch_size=16
