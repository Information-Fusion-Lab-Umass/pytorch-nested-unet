import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()
        loss = nn.MSELoss(reduction='mean')
        #print(input.size())
        #print(target.size())
        #exit()
        return loss(input, target)

class MSEAndBCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input0 = input[0]
        input1 = input[1]
        target0 = target[0]
        target1 = target[1]

        # BCEDice
        bce = F.binary_cross_entropy_with_logits(input0, target0)
        smooth = 1e-5
        input0 = torch.sigmoid(input0)
        num = target0.size(0)
        input0 = input0.view(num, -1)
        target0 = target0.view(num, -1)
        intersection = (input0 * target0)
        dice = (2. * intersection.sum(1) + smooth) / (input0.sum(1) + target0.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        bcediceloss = 0.5 * bce + dice

        # MSE
        input1 = input1.float()
        target1 = target1.float()
        input1 = input1.squeeze()
        target1 = target1.squeeze()
        loss = nn.MSELoss(reduction='mean')
        mseloss = loss(input1, target1)

        # Split the loss between segmentation and area
        return (bcediceloss * 0.5) + (mseloss * 0.5)