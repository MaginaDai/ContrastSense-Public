import argparse
from copy import deepcopy
import os

import torch
from torchvision import models

from ContrastSense import ContrastSense_model, ContrastSense
from data_loader.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_preprocessing.data_split import ClassesNum, UsersNum
from getPenalty import load_fisher_matrix
from utils import ContrastSense_evaluate, seed_torch
from torchvision.transforms import transforms
from data_loader import imu_transforms, emg_transforms

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ContrastSense for Wearable Sensing')
    
    parser.add_argument('-name', default='Myo', help='datasets name', 
                        choices=['HHAR', 'MotionSense', 'Shoaib', 'HASC', 'Myo', 'NinaPro', "Merged_dataset"])
    
    parser.add_argument('--pretrained', default='test/Myo', type=str, help='path to ContrastSense pretrained checkpoint')
    parser.add_argument('-version', default="shot0", type=str, help='control the version of the setting')

    parser.add_argument('-ft', '--if-fine-tune', default=True, type=bool, help='to decide whether tune all the layers')
    parser.add_argument('-shot', default=10, type=int, help='how many shots of labels to use')

    parser.add_argument('--store', default='test', type=str, help='define the name head for model storing')

    parser.add_argument('-e', '--epochs', default=4, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=256, type=int, help='feature dimension (default: 256)')

    parser.add_argument('--log-every-n-steps', default=4, type=int, help='Log every n steps')
    parser.add_argument('-t', '--temperature', default=1, type=float, help='softmax temperature (default: 1)')

    parser.add_argument('-g', '--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--evaluate', default=False, type=bool, help='To decide whether to evaluate')
    parser.add_argument('--resume', default='', type=str, help='To restart the model from a previous model')

    parser.add_argument('--mol', default='ContrastSense', type=str, help='which model to use', choices=['SimCLR', 'LIMU', 'CPC', 'ContrastSense'])

    parser.add_argument('-d', default=32, type=int, help='how many dims for encoder')

    parser.add_argument('-cd', '--classifer_dims', default=1024, type=int, help='the feature dims of the classifier')
    parser.add_argument('-final_dim', default=8, type=int, help='the output dims of the GRU')
    parser.add_argument('-mo', default=0.9, type=float, help='the momentum for Batch Normalization')

    parser.add_argument('-drop', default=0.1, type=float, help='the dropout portion')
    

    parser.add_argument('-penalty', default=True, type=bool, help='Use penalty or not')
    parser.add_argument('-penalty_lambda', default=50, type=float, help='penalty para')
    parser.add_argument('-penalty_pt', default=True, type=bool, help='use penalty acquired from pretrain or not')
    parser.add_argument('-fishermax', default=1e-2, type=float, help='fishermax')

    parser.add_argument('-cross', default='users', type=str, help='decide to use which kind of labels')
    parser.add_argument('-pretrain_lr', default=1e-4, type=float, help='learning rate during pretraining')
    parser.add_argument('-label_type', default=1, type=int, help='How many different kinds of labels for pretraining')

    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    if args.name in ['NinaPro', 'Myo', 'UCI']:
        args.modal = 'emg'
    else:
        args.modal = 'imu'
    
    args.transfer = True
    if args.store == None:
        args.store = deepcopy(args.pretrained)
    
    args.pretrained_model = './runs/' + args.pretrained + '/model_best.pth.tar'
    
    if args.penalty:
        if args.name == 'HHAR':
            args.fishermax = 1e-4
    
    if args.cross == "datasets":
        args.num_of_class = 4  # use the common dataset class name
    else:
        args.num_of_class = ClassesNum[args.name]

    return args
    
def main(args, fisher=None):
    
    dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=args.name, modal=args.modal)
    
    if args.modal == 'imu':
        tune_dataset = Dataset4Training(args.name, args.version, transform=transforms.Compose([imu_transforms.ToTensor()]), 
                                        split='tune', transfer=True, shot=args.shot, modal=args.modal)
    elif args.modal == 'emg':
        tune_dataset = Dataset4Training(args.name, args.version, transform=transforms.Compose([emg_transforms.EMGToTensor()]), 
                                        split='tune', transfer=True, shot=args.shot, modal=args.modal)
    else:
        NotADirectoryError

    val_dataset = dataset.get_dataset('val')
    test_dataset = dataset.get_dataset('test')
    

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, drop_last=False)


    tune_loader = torch.utils.data.DataLoader(
        tune_dataset, batch_size=int(args.batch_size), shuffle=True, pin_memory=False, drop_last=False)
    
    
    model = ContrastSense_model(transfer=True, classes=args.num_of_class, dims=args.d, 
                    classifier_dim=args.classifer_dims, final_dim=args.final_dim, 
                    momentum=args.mo, drop=args.drop, modal=args.modal)

    classifier_name = []
    # load pre-trained model
    for name, param in model.named_parameters():
        if "classifier" in name or 'discriminator' in name:
            classifier_name.append(name)
    
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            checkpoint = torch.load(args.pretrained_model, map_location="cpu")
            state_dict = checkpoint['state_dict']

            # rename ContrastSense pre-trained keys
            if args.mol == 'ContrastSense':
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q'):
                        # remove prefix
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            args.start_epoch = 0
            log = model.load_state_dict(state_dict, strict=False)
            if not args.evaluate:
                assert log.missing_keys == classifier_name
            print("=> loaded pre-trained model '{}'".format(args.pretrained_model))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_model))

    if not args.if_fine_tune:
        # froze the parameter
        for name, param in model.named_parameters():
            if name not in classifier_name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch'] + 1
            try:
                best_f1 = checkpoint['best_f1']
            except AttributeError:
                best_f1 = checkpoint['acc']
            best_f1 = torch.tensor(best_f1).to(args.device)  # best_acc1 may be from a checkpoint from a different GPU
            args.best_f1 = best_f1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not args.if_fine_tune:
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        if args.mol == "CPC" or args.mol == 'ContrastSense':
            assert len(parameters) == len(classifier_name)
        else:
            assert len(parameters) == 2 

    with torch.cuda.device(args.gpu_index):
        contrast = ContrastSense(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.evaluate:
            test_acc, test_f1 = ContrastSense_evaluate(model=contrast.model, criterion=contrast.criterion, args=contrast.args, data_loader=test_loader)
            print('test f1: {}'.format('%.3f' % test_f1))
            print('test acc: {}'.format('%.3f' % test_acc))
            return
        
        contrast.transfer_train_penalty(tune_loader, val_loader, fisher)
        best_model_dir = os.path.join(contrast.writer.log_dir, 'model_best.pth.tar')
        contrast.test_performance(best_model_dir=best_model_dir, test_loader=test_loader)
    
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed_torch()
    args = get_args()

    if args.penalty:
        fisher_cdl, fisher_infoNCE = load_fisher_matrix(args.pretrained, args.device)
        
        fisher = {n: torch.zeros(p.shape).to(args.device) for n, p in fisher_cdl.items()}

        for n, p in fisher_cdl.items():
            fisher_cdl[n] = torch.min(fisher_cdl[n], torch.tensor(args.fishermax)).to(args.device)

        fisher = fisher_cdl

    else:
        fisher = None

    main(args, fisher)
