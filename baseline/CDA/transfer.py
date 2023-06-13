import argparse
import sys
import os
import torch
from copy import deepcopy
from os.path import dirname

from baseline.CPCHAR.dataload import CPCHAR_Dataset
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from baseline.CDA.model import STCN
from baseline.CDA.ConSSL import ConSSL
from utils import evaluate, seed_torch
from data_aug.preprocessing import ClassesNum
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
parser.add_argument('-lr', '--learning-rate', default=5e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--transfer', default=True, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='CDA_HHAR', type=str, help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')

parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR'])
parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=3, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--kernel_size', type=int, default=3, help='Size of the conv filters in the encoder')

parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default=None, type=str, help='define the name head for model storing')
parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

parser.add_argument('--evaluate', default=False, type=bool, help='decide whether to evaluate')
parser.add_argument('-ft', '--if-fine-tune', default=False, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-version', default="shot", type=str, help='control the version of the setting')


def main():
    args = parser.parse_args()

    seed_torch(seed=args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    args.input_size = 6
    
    args.transfer = True

    if args.store == None:
        args.store = deepcopy(args.pretrained)
    
    args.pretrained = './runs/' + args.pretrained + '/model_best.pth.tar'

    dataset = CPCHAR_Dataset(transfer=True, version=args.version, datasets_name=args.name)
    tune_dataset = dataset.get_dataset('tune', percent=args.percent, shot=args.shot)
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    model = STCN(classes=ClassesNum[args.name], transfer=False)

    classifier_name = []
    for name, param in model.named_parameters():
        if "Classifier" in name:
            classifier_name.append(name)
    
    # load pre-trained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            args.start_epoch = 0
            log = model.load_state_dict(state_dict, strict=False)
            if not args.evaluate:
                assert log.missing_keys == classifier_name
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-6)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        cda = ConSSL(model=model, optimizer=optimizer, args=args)
        if args.evaluate:
            test_acc, test_f1 = evaluate(model=cda.model, criterion=cda.criterion, args=cda.args, data_loader=test_loader)
            print('test f1: {}'.format('%.3f' % test_f1))
            print('test acc: {}'.format('%.3f' % test_acc))
            return
    
        cda.transfer_train(tune_loader, val_loader)
        best_model_dir = os.path.join(cda.writer.log_dir, 'model_best.pth.tar')
        cda.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
