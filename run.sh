#!/bin/sh
python areaTrain.py --dataset long_axis_resize --arch NestedUNet --num_classes=2   --loss MSEAndBCEDiceLoss --lr=5e-9
