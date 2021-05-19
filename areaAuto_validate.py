import argparse
import os
from collections import OrderedDict
from glob import glob

import cv2
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

#import archs
import areaArchs
import losses
from areaDataset import areaDataset
from metrics import iou_score
from metrics import dice_coef
from sklearn.metrics import mean_squared_error
from utils import AverageMeter, str2bool

import numpy as np
import matplotlib.pyplot as plt

import random
from itertools import chain

#ARCH_NAMES = archs.__all__
ARCH_NAMES = areaArchs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
LOSS_NAMES.append('MSELoss')
LOSS_NAMES.append('MSEAndBCEDiceLoss')




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'mse': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, maskArea, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        maskArea = maskArea.cuda()
        #target = target.detach().numpy()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            #iou = iou_score(outputs[-1], target)
            #dice = dice_coef(outputs[-1], target)
        else:
            segMask, areaCalc = model(input)
            segMask = segMask.cuda()
            areaCalc = areaCalc.cuda()
            loss = criterion(segMask, areaCalc, target, maskArea, epoch)
            iou = iou_score(segMask, target)
            dice = dice_coef(segMask, target)
            #mse = mean_squared_error(output[1].detach().numpy(), target[1].detach().numpy())
            lossMSE = nn.MSELoss(reduction='mean')
            mse = lossMSE(areaCalc, maskArea)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))
        avg_meters['mse'].update(mse, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
            ('mse', avg_meters['mse'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('mse', avg_meters['mse'].avg)])


def validate(config, val_loader, model, criterion, epoch):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'mse': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, maskArea, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            maskArea = maskArea.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                #iou = iou_score(outputs[-1], target)
                #dice = dice_coef(outputs[-1], target)
            else:
                segMask, areaCalc = model(input)
                segMask = segMask.cuda()
                areaCalc = areaCalc.cuda()
                loss = criterion(segMask, areaCalc, target, maskArea, epoch)
                lossMSE = nn.MSELoss(reduction='mean')
                mse = lossMSE(areaCalc, maskArea)
                
                iou = iou_score(segMask, target)
                dice = dice_coef(segMask, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['mse'].update(mse, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('mse', avg_meters['mse'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg), 
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('mse', avg_meters['mse'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main_func(train_idx, val_set, test_set, modelName, fileName):
    config = vars(parse_args())
    config['name'] = modelName
    fw = open('batch_results_train/'+ fileName, 'w')
    print('config of dataset is ' + str(config['dataset']))
    fw.write('config of dataset is ' + str(config['dataset']) + '\n')    

    f = open('ids.txt', 'r')
    lines = f.readlines()

    lookup = []
    for i in range(len(lines)):
        lookup.append(lines[i].strip())

    val_idx = []
    for num in val_set:
        val_idx.append(lookup[num - 1])

    test_idx = []
    for num in test_set:
        test_idx.append(lookup[num - 1])

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = []
    train_img_ids = []

    for image in img_ids:
        im_begin = image.split('_')[0]
        if im_begin in val_idx:
            val_img_ids.append(image)
        elif im_begin not in test_idx:
            train_img_ids.append(image)

    print('config of dataset is ' + str(config['dataset']))
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    print('config loss is ' + str(config['loss']))
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
        #criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()
        #criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    #model = archs.__dict__[config['arch']](config['num_classes'],
    #                                       config['input_channels'],
    #                                       config['deep_supervision'])
    model = areaArchs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    train_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    train_dataset = areaDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = areaDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('mse', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []), 
        ('val_mse', []),
    ])

    #best_iou = 0
    trigger = 0
    #best_dice = 0
    best_mse = float("inf")
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        fw.write('Epoch [%d/%d]' % (epoch, config['epochs']) + '\n')    

        # Freeze Segmentation Layers, Update model, optimizer, and scheduler
        if (epoch == 25):
            # Freeze all but the regression (area) layers
            for param in model.parameters():
                param.requires_grad = False
            for param in model.regression.parameters():
                param.requires_grad = True

            # Update the optimizer to only include the regression layers
            if config['optimizer'] == 'Adam':
                optimizer = optim.Adam(
                    model.regression.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            elif config['optimizer'] == 'SGD':
                optimizer = optim.SGD(model.regression.parameters(), lr=config['lr'], momentum=config['momentum'],
                                    nesterov=config['nesterov'], weight_decay=config['weight_decay'])
            else:
                raise NotImplementedError

            # Update the scheduler to include the new optimizer
            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                        verbose=1, min_lr=config['min_lr'])
            elif config['scheduler'] == 'MultiStepLR':
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
            elif config['scheduler'] == 'ConstantLR':
                scheduler = None
            else:
                raise NotImplementedError

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, epoch)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - mse %.4f - val_loss %.4f - val_iou %.4f val_dice %.4f - val_mse %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], train_log['mse'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['mse']))
        fw.write('loss %.4f - iou %.4f - dice %.4f - mse %.4f - val_loss %.4f - val_iou %.4f val_dice %.4f - val_mse %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], train_log['mse'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['mse']) + '\n')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['mse'].append(train_log['mse'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_mse'].append(val_log['mse'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1
        '''
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        '''
        if val_log['mse'] < best_mse:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_mse = val_log['mse']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            fw.write("=> early stopping" + '\n')
            break

        torch.cuda.empty_cache()

    # Create Plots

    # Segmentation Plot
    x_axis = np.arange(0,config['epochs']-5)
    plt.plot(x_axis, log['loss'][0:25], label="train_loss", color="r")
    plt.plot(x_axis, log['iou'][0:25], label="train_iou", color="r", linestyle=":")
    plt.plot(x_axis, log['dice'][0:25], label="train_dice", color="r", linestyle="--")
    plt.plot(x_axis, log['val_loss'][0:25], label="val_loss", color="b")
    plt.plot(x_axis, log['val_iou'][0:25], label="val_iou", color="b", linestyle=":")
    plt.plot(x_axis, log['val_dice'][0:25], label="val_dice", color="b", linestyle="--")
    plt.xlabel("Epoch")
    plt.title("Segmentation - {}".format(config['name']))
    plt.legend()

    plt.savefig('models/%s/seg_plot.png' % config['name'], dpi=300, bbox_inches='tight')

    plt.clf()

    # Area Loss Plot
    x_axis = np.arange(config['epochs']-5,config['epochs'])
    plt.plot(x_axis, log['loss'][25:30], label="train_loss", color="r")
    #plt.plot(x_axis, log['mse'][25:30], label="train_mse", color="r")
    plt.plot(x_axis, log['val_loss'][25:30], label="val_loss", color="b")
    #plt.plot(x_axis, log['val_mse'][25:30], label="val_mse", color="b")
    plt.xlabel("Epoch")
    plt.title("Area Loss - {}".format(config['name']))
    plt.legend()

    plt.savefig('models/%s/area_loss_plot.png' % config['name'], dpi=300, bbox_inches='tight')

    plt.clf()

    # Area Loss Plot
    x_axis = np.arange(config['epochs']-5,config['epochs'])
    #plt.plot(x_axis, log['loss'][25:30], label="train_loss", color="r")
    plt.plot(x_axis, log['mse'][25:30], label="train_mse", color="r")
    #plt.plot(x_axis, log['val_loss'][25:30], label="val_loss", color="b")
    plt.plot(x_axis, log['val_mse'][25:30], label="val_mse", color="b")
    plt.xlabel("Epoch")
    plt.title("Area MSE - {}".format(config['name']))
    plt.legend()

    plt.savefig('models/%s/area_mse_plot.png' % config['name'], dpi=300, bbox_inches='tight')

    plt.clf()

def perform_validation(modelName, testNum, fileName):
    #args = parse_args()

    fw = open('batch_results_test/' + fileName, 'w') 
    #with open('models/%s/config.yml' % args.name, 'r') as f:
    with open('models/%s/config.yml' % modelName, 'r') as f:   
        config = yaml.load(f, Loader=yaml.FullLoader)
 
    #config['dataset'] = 'ax_crop_val_' + str(testNum) + '_' + str(testNum + 1)

    print('-'*20)
    fw.write('-'*20 + '\n')
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
        fw.write('%s: %s' % (key, str(config[key])) + '\n')
    print('-'*20)
    fw.write('-'*20 + '\n')

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')
    model = areaArchs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    f = open('ids.txt', 'r')
    lines = f.readlines()

    lookup = []
    for i in range(len(lines)):
        lookup.append(lines[i].strip())

    test_idx = []
    for num in testNum:
        test_idx.append(lookup[num - 1])
    
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    test_img_ids = []
    for img in img_ids:
        im_begin = img.split('_')[0]
        if im_begin in test_idx:
            test_img_ids.append(img)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_dataset = areaDataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    mse_avg_meter = AverageMeter()

    #for c in range(config['num_classes']):
    os.makedirs(os.path.join('outputs', config['name'], str(1)), exist_ok=True)
    with torch.no_grad():
        for input, target, maskArea, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()
            maskArea = maskArea.cuda()

            # compute output
            if config['deep_supervision']: # Not currently used
                output = model(input)[-1]
            else:
                segMask, areaCalc = model(input)

            iou = iou_score(segMask, target)
            iou_avg_meter.update(iou, input.size(0))

            dice = dice_coef(segMask, target)
            dice_avg_meter.update(dice, input.size(0))

            lossMSE = nn.MSELoss(reduction='mean')
            mse = lossMSE(areaCalc, maskArea)
            mse_avg_meter.update(mse, input.size(0))

            #output = torch.sigmoid(output).cpu().numpy()

            #for i in range(len(output)):
            #    for c in range(config['num_classes']):
            #        cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
            #                    (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    fw.write('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    fw.write('Dice: %.4f' % dice_avg_meter.avg)
    print('MSE: %.4f' % mse_avg_meter.avg)
    fw.write('MSE: %.4f' % mse_avg_meter.avg)

    torch.cuda.empty_cache()

def main():
    '''params = {}
    params['dataset'] = 'sa_dataset'
    params['loss'] = 'BCEDiceLoss'
    params['arch'] = 'NestedUNet'
    params['num_classes'] = 2
    params['input_channels'] = 3
    params['deep_supervision'] = False
    params['optimizer'] = 'SGD'
    params['lr'] = 1e-3
    params['weight_decay'] = 1e-4
    params['momentum'] = 0.9
    params['nesterov'] = False
    params['scheduler'] = 'CosineAnnealingLR'
    params['img_ext'] = 'png'
    params['mask_ext'] = 'png'
    params['input_h'] = 96   ## can be set to a command line argument in the future
    params['input_w'] = 96   ## can be set to a command line argument in the future
    params['batch_size'] = 16
    params['num_workers'] = 4
    params['epochs'] = 100
    params['early_stopping'] = -1
    params['min_lr'] = 1e-5
    # extras
    params['factor'] = 0.1
    params['patience'] = 2
    params['milestones'] = '1,2'
    params['gamma'] = 0.66666
    '''
    #params = vars(parse_args())
    for i in range(0, 100, 20):
        for j in range(0, 100, 20):
            if j == i:
                continue
            use = []
            for k in range(0, 100, 20):
                if k == i or k == j:
                    continue
                use.append(k)
            train_set = chain(range(use[0], use[0]+20), range(use[1], use[1]+20), range(use[2], use[2]+20))
            val_set = range(j, j+20)
            test_set = range(i, i+20)
            modelName = 'long_axis_resize_batch_' + str(i) + '_' + str(i + 20) + '_test_' + str(j) + '_' + str(j + 20) + '_val'
            trainFileName = 'long_axis_resize_batch_' + str(i) + '_' + str(i + 20) + '_test_' + str(j) + '_' + str(j + 20) + '_val_' + '_trainingResult'
            testFileName = 'long_axis_resize_batch_' + str(i) + '_' + str(i + 20) + '_test_' + str(j) + '_' + str(j + 20) + '_val_' + '_testResult'
            main_func(train_set, val_set, test_set, modelName, trainFileName)
            perform_validation(modelName, test_set, testFileName)

if __name__ == '__main__':
    main()
