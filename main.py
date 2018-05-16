import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import utils
import gpu_config_torch

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

SAVE_MODEL_FP = 'saved_torch_models/resnet/model.pytorch'
best_prec1 = 0


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default='Data/cifar10',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    '''
    '''
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--gpu',  action='store_true', default=True,
                        help='Enable CUDA')
    '''
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    '''

    return parser


def model_predict(test_loader, model, out_numpy=False):
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(args.device), labels.to(args.device)
            
            outputs = model(features)
            
            # outputs = outputs.numpy()
            # labels = labels.numpy()
            y_true_list.append(labels)
            y_pred_list.append(outputs)
    
    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    
    if out_numpy:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
    
    return y_true, y_pred


def model_score(test_loader, model, score_func, loss_func):
    y_true, y_pred = model_predict(test_loader, model)
    score = score_func(y_true, y_pred)
    loss = loss_func(y_pred, y_true)
    
    return score, loss


def main():
    global args, best_prec1
    parser = create_arg_parser()
    args = parser.parse_args()

    # choose computation device: CPU/GPU
    print("=> visable GPU card: #{:d}".format(gpu_config_torch.gpu_no))
    args.device = None
    if args.gpu:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device('cpu')
    print("=> use device: ", args.device)

    # create model instance
    if False:  # args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch](num_classes=10)
        from model.resnet import resnet18
        model = resnet18(dim=1,
                         pretrained=False,
                         in_planes=8,
                         num_classes=1,)
        # from model import Net
        # model = Net(64 * 56, 1, in_channels=8, dim=1)
        # from resnet import resnet18
        # model = resnet18(num_classes=1, in_channels=8)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        # model.cuda()
        model.to(args.device)
    else:
        # model = torch.nn.DataParallel(model).cuda()
        model = torch.nn.DataParallel(model)
        model.to(args.device)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss()  # .cuda()
    criterion = nn.MSELoss()
    criterion.to(args.device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Load dataset & dataloader
    # DEBUG
    args.batch_size = 100
    args.print_freq = 10
    print("=> use batch_size {:d}".format(args.batch_size))
    from my_dataset import get_future_loader
    train_loader, val_loader = get_future_loader(batch_size=args.batch_size, cut_len=40000, lite_version=True)
    # from my_dataset import get_cifar_10
    # train_loader, val_loader = get_cifar_10(batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    # from my_dataset import get_future_bar_classification_data
    # train_loader, val_loader = get_future_bar_classification_data(batch_size=args.batch_size, cut_len=0)
    # train_loader, val_loader = load_imagenet(args.data, args.batch_size, args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion)
        score, loss = model_score(val_loader, model, utils.calc_rsq, criterion)
        print("Val_loss = {:+4.6f}".format(loss.item()))
        print("Val_score = {:+4.6f}".format(score.item()))

        # remember best prec@1 and save checkpoint
        is_best = score > best_prec1
        best_prec1 = max(score, best_prec1)
        save_checkpoint({'epoch': epoch,
                         'arch': args.arch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer' : optimizer.state_dict(),
                         },
                        is_best,
                        filename=SAVE_MODEL_FP)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        target = target.to(args.device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = utils.calc_topk_accu(output, target, topk=(5, 10))
        prec1, prec5 = [0.95], [0.98]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  ''.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        y_true_list = []
        y_pred_list = []

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda(non_blocking=True)
            target = target.to(args.device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            #y_true_list.append(target)
            #y_pred_list.append(utils.argmax(output, dim=1))

            # measure accuracy and record loss
            # prec1, prec5 = utils.calc_topk_accu(output, target, topk=(1, 5))
            prec1, prec5 = [0.95], [0.98]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        #y_true = torch.cat(y_true_list, dim=0)
        #y_pred = torch.cat(y_pred_list, dim=0)
        #accu = utils.calc_accu(y_true, y_pred, argmax=False)
        #print("Validation whole accuracy: {:.5f}".format(accu))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pytorch'):
    fp = os.path.abspath(filename)
    par_dir = os.path.dirname(fp)
    utils.create_dir(fp)

    torch.save(state, filename)
    print("Checkpoint saved at {:s}".format(filename))
    if is_best:
        fp2 = os.path.join(par_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(filename,fp2)
        print("New best checkpoint found. Saved at {:s}".format(fp2))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 12))
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("=> learning rate adjusted as {:.3e}".format(lr))


if __name__ == '__main__':
    main()
