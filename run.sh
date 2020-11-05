#!/bin/sh
python train.py --dataset axial_full_train --arch NestedUNet --img_ext .png --mask_ext .png --num_classes=2 --input_w=96 --input_h=96
