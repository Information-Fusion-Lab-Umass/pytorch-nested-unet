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

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from metrics import dice_coef
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()

    # model name, used for storing
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    # architectures defined in archs.py
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')

    # kept False always
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

    # dataset name which must be stored in datasets/ directory
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
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 
                                 'MultiStepLR', 'ConstantLR'])
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

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to training mode
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def main_func(train_idx, val_set, test_set, modelName, fileName):
    '''
        params: train_idx, val_set, test_set => patient ids in train, val and test set.
                modelName, fileName => modelname for model directory storing models, 
                configurations and filename to store results, both generated as per 
                patient indices in train, test and val set. (For identification later)
        New model trained, tested and stored in corresponding modelName and fileName files.
        No objects returned.
    '''
    
    # Read configurations and create model directory
    config = vars(parse_args())
    config['name'] = modelName
    fw = open('batch_results_train/'+ fileName, 'w')
    print('config of dataset is ' + str(config['dataset']))
    fw.write('config of dataset is ' + str(config['dataset']) + '\n')    
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    fw.write('-' * 20 + '\n')
    for key in config:
        print('%s: %s' % (key, config[key]))
        fw.write('%s: %s' % (key, config[key]) + '\n')
    print('-' * 20)
    fw.write('-' * 20 + '\n')

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')   
    model = archs.__dict__[config['arch']](config['num_classes'],
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

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # Patient IDs in validation set
    val_idx = [val_set]

    # Creating lists for noting images in train set or validation set. Iterate 
    # through all images and insert into train or validation set as per patient
    # found in the name of the image.
    val_img_ids = []
    train_img_ids = []

    for image in img_ids:
        im_begin = image.split('.')[0]
        if int(im_begin[-1]) in val_idx:
            val_img_ids.append(image)
        elif int(im_begin[-1]) in train_idx:
            train_img_ids.append(image)

    # Transformations that could be applied to the images.
    # Note: Same transformation must be applied to both train and validation set.
    train_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    train_transform2 = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    val_transform2 = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
        transforms.ShiftScaleRotate(shift_limit = 0.1, scale_limit = 0, rotate_limit = 0),# shift_limit_x = 0.1, shift_limit_y = 0.1, p = 1), ##TODO remove from validation
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    
    # Creating PyTorch dataset object.
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform2)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    
    # creating the pytorch dataloader for train and validation sets.
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

    # Results dictionary
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('dice', []),
    ])

    best_iou = 0
    trigger = 0
    best_dice = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        fw.write('Epoch [%d/%d]' % (epoch, config['epochs']) + '\n')    

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice']))
        fw.write('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice']) + '\n')

        # Appending result to log dictionary
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        # Determine if new updated model gives best performance and accordingly save.
        # Multiple ways to determine better performance, dice score is used here, can also use IoU.
        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['dice']
            print("=> saved best model")
            fw.write("=> saved best model" + '\n')
            trigger = 0

        '''
        # can be used if best model picked using IOU
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        '''

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            fw.write("=> early stopping" + '\n')
            break

        torch.cuda.empty_cache()

def perform_validation(modelName, testNum, fileName):
    '''
        params: modelName, fileName => modelname for loading models from model directory, 
                and filename to store results, both generated as per patient indices in 
                train, test and val set. (For identification later)
                testNum => patient indices in test set
        Trained model tested on test set and results stored in fileName.
        No objects returned.
    '''
    fw = open('batch_results_val/' + fileName, 'w') 
    with open('models/%s/config.yml' % modelName, 'r') as f:   
        config = yaml.load(f, Loader=yaml.FullLoader)

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
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 2 patients data used for validation. Filtering those images from the 
    # entire dataset.
    val_idx = [testNum, testNum + 1]
    val_img_ids = []
    for img in img_ids:
        im_begin = img.split('.')[0]
        if int(im_begin[-1]) in val_idx:
            val_img_ids.append(img)

    # Loading model and setting to evaluation model (since we only need forward pass)
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    # Pytorch objects for transformation, dataset and dataloader
    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    # Running forward pass and storing results including masks, IoU and dice score.
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            dice = dice_coef(output, target)
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meter.avg)
    fw.write('IoU: %.4f' % avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    fw.write('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()

# Total 10 patients in HVSMR dataset of which we use first 8 for cross validation
# having patient IDs(0-7). Following code creates combinations having 2 patients 
# in test set, 4 or 5 of the rest for training the model and rest 1 or 2 for 
# validation. To prevent an exponential growth in number of combinations, test set 
# patient id's are required to be consective. (i.e. test set will have only the 
# following combinations - (0, 1), (2, 3), (4, 5), (6, 7))
def main():
    for i in range(0, 8, 2):
        for j in range(0, 8, 1):
            if j == i or j == i + 1:
                continue
            use = []
            for k in range(8):
                if k == i or k == j or k == i + 1:
                    continue
                use.append(k)
            
            modelName = 'shortAxis_1val_batch_' + str(i) + '_' + str(i + 1) + '_test_' + str(j) + '_val'
            trainFileName = 'shortAxis_1val_batch_' + str(i) + '_' + str(i + 1) + '_test_' + str(j) + '_val_' + '_trainingResult'
            valFileName = 'shortAxis_1val_batch_' + str(i) + '_' + str(i + 1) + '_test_' + str(j) + '_val_' + '_validationResult'
            main_func(use, j, i, modelName, trainFileName)
            perform_validation(modelName, i, valFileName)

if __name__ == '__main__':
    main()
