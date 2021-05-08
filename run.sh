#!/bin/sh
python areaTrain.py --dataset long_axis_resize --arch NestedUNet --num_classes=1  --loss MSEAndBCEDiceLoss --lr=5e-4 --name="long_axis_resize_with_lr_5e_4" --epoch=30
