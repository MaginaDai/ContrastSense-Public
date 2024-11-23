import os
import random
import numpy as np
import torch
import yaml
from data_loader.contrastive_learning_dataset import ContrastiveLearningDataset, fetch_dataset_root
from data_preprocessing.data_split import UsersPosition
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    return


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        # shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
        shape[shape.index(-1)] = torch.div(x.size(-1), -np.prod(shape), rounding_mode='trunc') 
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', path_dir='./'):
    torch.save(state, filename)  # directly save the best model
    # if is_best:
    #     shutil.copyfile(filename, os.path.join(path_dir, 'model_best.pth.tar'))


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res = correct_k.mul_(100.0 / batch_size)
        
        return res.cpu().numpy()[0]

def f1_cal(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)

        f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='macro') * 100
    
    return f1


def evaluate(model, criterion, args, data_loader):
    losses = AverageMeter('Loss', ':.4e')
    acc_eval = AverageMeter('acc_eval', ':6.2f')
    # f1_eval = AverageMeter('f1_eval', ':6.2f')

    model.eval()

    label_all = []
    pred_all = []

    with torch.no_grad():
        for sensor, target in data_loader:
            sensor = sensor.to(args.device)
            if sensor.shape == 2:
                sensor = sensor.unsqueeze(dim=0)
            target = target[:, 0].to(args.device)

            with autocast(enabled=args.fp16_precision):
                logits = model(sensor)
                if type(logits) is tuple:
                    logits = logits[0]
                loss = criterion(logits, target)

            losses.update(loss.item(), sensor.size(0))
            _, pred = logits.topk(1, 1, True, True)

            label_all = np.concatenate((label_all, target.cpu().numpy()))
            pred_all = np.concatenate((pred_all, pred.cpu().numpy().reshape(-1)))

            acc = accuracy(logits, target, topk=(1,))
            acc_eval.update(acc, sensor.size(0))

    f1_eval = f1_score(label_all, pred_all, average='macro') * 100

    return acc_eval.avg, f1_eval


def ContrastSense_evaluate(model, criterion, args, data_loader, test=False):
    losses = AverageMeter('Loss', ':.4e')
    acc_eval = AverageMeter('acc_eval', ':6.2f')

    model.eval()
    
    label_all = []
    pred_all = []

    with torch.no_grad():
        for sensor, target in data_loader:
            sensor = sensor.to(args.device)
            if sensor.shape == 2:
                sensor = sensor.unsqueeze(dim=0)
            target = target[:, 0].to(args.device)

            with autocast(enabled=args.fp16_precision):
                logits = model(sensor)
                loss = criterion(logits, target)

            losses.update(loss.item(), sensor.size(0))
            _, pred = logits.topk(1, 1, True, True)
            
            label_all = np.concatenate((label_all, target.cpu().numpy()))
            pred_all = np.concatenate((pred_all, pred.cpu().numpy().reshape(-1)))

            acc = accuracy(logits, target, topk=(1,))
            acc_eval.update(acc, sensor.size(0))

    f1_eval = f1_score(label_all, pred_all, average='macro') * 100
    # if test:
    #     np.savez('pred_all.npz', pred_all=pred_all)
    return acc_eval.avg, f1_eval


def CPC_evaluate(model, criterion, args, data_loader):
    losses = AverageMeter('Loss', ':.4e')
    acc1 = AverageMeter('acc1', ':6.2f')

    model.eval()

    with torch.no_grad():
        for sensor, target in data_loader:
            sensor = sensor.to(args.device)
            target = target.to(args.device)

            with autocast(enabled=args.fp16_precision):
                logits = model(sensor)
                loss = criterion(logits, target)

            losses.update(loss.item(), sensor.size(0))
            _, pred = logits.topk(1, 1, True, True)
            acc = accuracy(logits, target, topk=(1,))
            acc1.update(acc, sensor.size(0))

    return acc1.avg

def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    return device

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def identify_users_number(version, dataset):
    ori_dir = fetch_dataset_root(dataset)
    dir = ori_dir + '_' + version + '/train_set.npz'
    data = np.load(dir)
    train_set = data['train_set']
    user = []
    for i in train_set:
        sub_dir = ori_dir + '/' + i
        data = np.load(sub_dir, allow_pickle=True)
        if dataset == 'Shoaib':
            user.append(int(data['add_infor'][0, UsersPosition[dataset]]))
        else:
            user.append(data['add_infor'][0, UsersPosition[dataset]])
    
    user_type = np.unique(user)
    return len(user_type)


def fetch_test_loader_for_all_dataset(args, datasets):
    test_loader_for_all_datasets = []
    for name in datasets:
        CL_dataset = ContrastiveLearningDataset(transfer=True, version=args.version, datasets_name=name, cross_dataset=True)
        test_dataset = CL_dataset.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=False)
        test_loader_for_all_datasets.append(test_loader)

    return test_loader_for_all_datasets


def compute_kernel(x, y, sigma):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    
    x = x.unsqueeze(1)  # Make it into a column tensor
    y = y.unsqueeze(0)  # Make it into a row tensor
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / (4 * sigma * sigma))

def compute_mmd(x, y, sigma):
    x_kernel = compute_kernel(x, x, sigma)
    y_kernel = compute_kernel(y, y, sigma)
    xy_kernel = compute_kernel(x, y, sigma)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

