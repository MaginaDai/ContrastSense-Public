import argparse
import sys
import os
import torch
from copy import deepcopy
from os.path import dirname

sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from torchvision.transforms import transforms
from CLISA.clisa import CLISA
from CLISA.model import CLISA_model
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_aug.preprocessing import ClassesNum
from utils import evaluate, seed_torch
from data_aug import emg_transforms
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--transfer', default=False, type=str, help='to tell whether we are doing transfer learning')
parser.add_argument('--pretrained', default='test/SEED', type=str, help='path to pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('-name', default='SEED', help='datasets name', choices=['SEED', 'SEED_IV'])

parser.add_argument('--log-every-n-steps', default=5, type=int, help='Log every n steps')
parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 5)')
parser.add_argument('--store', default=None, type=str, help='define the name head for model storing')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

parser.add_argument('--evaluate', default=False, type=bool, help='decide whether to evaluate')
parser.add_argument('-ft', '--if-fine-tune', default=True, type=bool, help='to decide whether tune all the layers')
parser.add_argument('-version', default="shot1", type=str, help='control the version of the setting')


def main():
    args = parser.parse_args()
    seed_torch(seed=args.seed)

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    args.transfer = True
    args.modal = 'eeg'

    if args.store == None:
        args.store = deepcopy(args.pretrained)
    
    args.pretrained = './runs/' + args.pretrained + '/model_best.pth.tar'

    dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=args.name, if_baseline=True, modal='eeg')
    tune_dataset = Dataset4Training(args.name, args.version, transform=transforms.Compose([emg_transforms.EMGToTensor()]), 
                                        split='tune', transfer=True, shot=args.shot, modal=args.modal, if_baseline=True)
    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')

    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)

    model = CLISA_model(transfer=True, num_class=ClassesNum[args.name])

    # model.to(args.device)

    classifier_name = []
    for name, param in model.named_parameters():
        if "classifier" in name:
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

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0.005)

    with torch.cuda.device(args.gpu_index):
        clisa = CLISA(model=model, optimizer=optimizer, scheduler=None, args=args)
        if args.evaluate:
            test_acc, test_f1 = evaluate(model=clisa.model, criterion=clisa.criterion, args=clisa.args, data_loader=test_loader)
            print('test f1: {}'.format('%.3f' % test_f1))
            print('test acc: {}'.format('%.3f' % test_acc))
            return
        clisa.transfer_train(tune_loader, val_loader)
        best_model_dir = os.path.join(clisa.writer.log_dir, 'model_best.pth.tar')
        clisa.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
