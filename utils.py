import argparse
import json
import os
import random
import shutil
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
from typing import NamedTuple
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, fetch_dataset_root
from data_aug.imu_transforms import ACT_Translated_labels, HHAR_movement
from data_aug.preprocessing import UsersPosition
from exceptions.exceptions import InvalidDatasetSelection
from judge import AverageMeter
from torch.cuda.amp import autocast
import pdb
from baseline.LIMU_BERT.config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config
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
    return


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
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
                loss = criterion(logits, target)

            losses.update(loss.item(), sensor.size(0))
            _, pred = logits.topk(1, 1, True, True)

            label_all = np.concatenate((label_all, target.cpu().numpy()))
            pred_all = np.concatenate((pred_all, pred.cpu().numpy().reshape(-1)))

            acc = accuracy(logits, target, topk=(1,))
            acc_eval.update(acc, sensor.size(0))

    f1_eval = f1_score(label_all, pred_all, average='macro') * 100

    return acc_eval.avg, f1_eval


def MoCo_evaluate(model, criterion, args, data_loader, test=False):
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
                logits, _ = model(sensor)
                loss = criterion(logits, target)

            losses.update(loss.item(), sensor.size(0))
            _, pred = logits.topk(1, 1, True, True)
            
            label_all = np.concatenate((label_all, target.cpu().numpy()))
            pred_all = np.concatenate((pred_all, pred.cpu().numpy().reshape(-1)))

            acc = accuracy(logits, target, topk=(1,))
            acc_eval.update(acc, sensor.size(0))

    f1_eval = f1_score(label_all, pred_all, average='macro') * 100
    if test:
        np.savez('pred_all.npz', pred_all=pred_all)
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


class PretrainModelConfig(NamedTuple):
    "Configuration for BERT model"
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    feature_num: int = 0  # Factorized embedding parameterization

    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = True

    @classmethod
    def from_json(cls, js):
        return cls(**js)


class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    input: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    return device


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0, dataset_name='HHAR'):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma
        self.dataset_name = dataset_name

    def __call__(self, sample):
        acc, gyro, add_infor = sample['acc'], sample['gyro'], sample['add_infor']
        if 'HHAR' in self.dataset_name:
            label = np.array([HHAR_movement.index(add_infor[0, -1])])
        elif self.dataset_name == 'MotionSense':
            label = np.array([ACT_Translated_labels.index(add_infor[0, -1])])
        elif self.dataset_name == 'UCI':
            label = np.array([int(add_infor[0, -2])])
        elif self.dataset_name == 'Shoaib':
            label = np.array([int(add_infor[0, -2])])
        elif self.dataset_name == 'HASC':
            label = np.array([int(add_infor[0, -1])])
        elif self.dataset_name == 'ICHAR':
            label = np.array([int(add_infor[0, -1])])
        else:
            raise InvalidDatasetSelection()
        instance = np.concatenate([acc, gyro], axis=1)
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        
        return instance_new, label


class Preprocess4Tensor(Pipeline):
    def __init__(self, data_type='float'):
        super().__init__()
        self.data_type = data_type

    def __call__(self, sample):
        instance, label = sample
        instance = instance.astype(np.float64)
        if self.data_type == 'float':
            return torch.from_numpy(instance).float(), torch.from_numpy(label).squeeze()
        else:
            raise NotImplementedError


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, data):
        instance, label = data
        instance = instance.astype(np.float64)
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        # pdb.set_trace()
        return torch.from_numpy(instance_mask.astype(np.float64)).float(), torch.from_numpy(np.array(mask_pos_index)), torch.from_numpy(np.array(seq)).float(), label


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    seed_torch(train_cfg.seed)
    return train_cfg, model_cfg, mask_cfg


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_version', type=str, help='Model config')
    parser.add_argument('dataset_version',  type=str, help='Dataset version')
    parser.add_argument('-g', '--gpu_index', default=0, type=int, help='Gpu index.')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model name')
    parser.add_argument('-p', '--pretrain_dataset', type=str, default=None, help='Pretrain dataset name')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model',
                        help='The saved model name')
    parser.add_argument('-name', default='HHAR', help='datasets name', choices=['HHAR', 'MotionSense', 'UCI', 'Shoaib', 'HASC', 'ICHAR'])

    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
    parser.add_argument('-percent', default=1, type=float, help='how much percent of labels to use')
    parser.add_argument('-shot', default=None, type=int, help='how many shots of labels to use')
    parser.add_argument('-frozen_bert', default=False, type=int, help='how many shots of labels to use')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    args = create_io_config(args, args.name, args.dataset_version, pretrain_model=args.model_file, target=target, model_file=args.pretrain_dataset)
    return args


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    return train_cfg, model_bert_cfg, model_classifier_cfg


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
