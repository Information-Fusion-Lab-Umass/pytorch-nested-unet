#!/bin/sh
python train.py --dataset short_axis_resize --dataset2 long_axis_resize --arch MultiViewNestedUNet --num_classes=2 --secondViewPath models/long_axis_single_view/model.pth
