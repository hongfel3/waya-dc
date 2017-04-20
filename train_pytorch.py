import os
import random
import time

from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import dataloader
from torchvision import models
from torchvision import transforms

from wayadc.utils import helpers
from wayadc.utils import image_generator


#
# directories and paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')
data_dir = os.path.join(path, 'data')

valid_dir = 'data-scraped-dermnetnz'
model_checkpoint = os.path.join(cache_dir, 'model_checkpoint.tar')

log_file_path = 'train_log.txt'

#
# image dimensions
#

img_height = 224
img_width = 224
img_channels = 3

#
# training params
#

batch_size = 96
nb_epoch = 150


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_batch, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_batch)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input_batch.size(0))
        top1.update(prec1[0], input_batch.size(0))
        top3.update(prec3[0], input_batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            st = ('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))

            with open(log_file_path, 'a') as f:
                f.write('{}\n'.format(st))

            print(st)


def valid(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_batch, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_batch, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input_batch.size(0))
        top1.update(prec1[0], input_batch.size(0))
        top3.update(prec3[0], input_batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            st = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top3=top3))

            with open(log_file_path, 'a') as f:
                f.write('{}\n'.format(st))

            print(st)

    st = (' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

    with open(log_file_path, 'a') as f:
        f.write('{}\n'.format(st))

    print(st)

    return top1.avg


def accuracy(output, target, topk=(1,)):
    batch_size = target.size(0)

    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def main():
    #
    # set up train and valid dirs
    #

    train_dirs = set()
    for dataset_dir in helpers.list_dir(data_dir, sub_dirs_only=True):
        if dataset_dir == valid_dir or not dataset_dir.startswith('data-scraped-'):
            continue

        train_dirs.add(os.path.join(data_dir, dataset_dir))

    valid_dirs = {os.path.join(data_dir, valid_dir)}
    assert not train_dirs.intersection(valid_dirs)

    print(train_dirs, valid_dirs)

    #
    # set up data loaders
    #

    train_transform = transforms.Compose([
        transforms.Scale(size=256, interpolation=Image.ANTIALIAS),
        transforms.RandomSizedCrop(size=img_height, interpolation=Image.ANTIALIAS),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambd=lambda im: im.transpose(Image.FLIP_TOP_BOTTOM) if random.random() < 0.5 else im),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Scale(size=256, interpolation=Image.ANTIALIAS),
        transforms.CenterCrop(size=img_height),
        transforms.ToTensor(),
    ])

    train_generator = image_generator.ImageGenerator(train_dirs, target_size=(img_height, img_width),
                                                     transformation_pipeline=train_transform)
    valid_generator = image_generator.ImageGenerator(valid_dirs, target_size=(img_height, img_width),
                                                     transformation_pipeline=valid_transform)
    assert train_generator._groups == valid_generator._groups

    train_loader = torch.utils.data.DataLoader(train_generator, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_generator, batch_size=batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)

    def get_class_weights(nb_samples_per_class):
        print('Number of samples per class: {}.'.format(nb_samples_per_class))

        weights = {}
        for cls, nb_samples in nb_samples_per_class.items():
            weights[cls] = nb_samples_per_class.get(0) / nb_samples

        _weights = []
        for cls in sorted(weights.keys()):
            _weights.append(weights.get(cls))

        return torch.FloatTensor(_weights)

    # our classes are imbalanced so we need `class_weights` to scale the loss appropriately
    class_weights = get_class_weights(train_generator.label_sizes)
    print(class_weights)

    #
    # set up model
    #

    model = models.resnet50(pretrained=True)
    nb_features = model.fc.in_features
    model.fc = nn.Linear(nb_features, len(train_generator._groups))

    for param in model.conv1.parameters():
        param.requires_grad = False

    for param in model.bn1.parameters():
        param.requires_grad = False

    for param in model.layer1.parameters():
        param.requires_grad = False

    for param in model.layer2.parameters():
        param.requires_grad = False

    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    cudnn.benchmark = True

    print(model)

    #
    # train
    #

    for epoch in range(nb_epoch):
        best_precision = 0

        train(train_loader, model, criterion, optimizer, epoch)
        precision = valid(valid_loader, model, criterion)

        if precision > best_precision:
            best_precision = precision

            torch.save({
                'epoch': epoch + 1,
                'model_architecture': 'resnet50',
                'model_state_dict': model.state_dict(),
                'precision': best_precision,
                'optimizer_state_dict': optimizer.state_dict()
            }, model_checkpoint)


if __name__ == '__main__':
    main()
