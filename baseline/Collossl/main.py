import datetime 

import sys
import torch
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))

from Collossl_framework import collossl_framework
from collossl_dataloader import ColloSSL_Dataset
from contrastive_training import get_group_held_out_users
from device_selection import get_pos_neg_apriori
from simclr_models_torch import TPN_encoder
from common_parser import get_parser
import os
import numpy as np

from utils import seed_torch

## Loss function import
from loss_fn import *


def main():
    parser = get_parser()

    ## Prepare full dataset
    args = parser.parse_args()

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu_index}')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    if args.eval_device is None:
        args.eval_device = args.train_device
    if args.fine_tune_device is None:
        args.fine_tune_device = args.train_device

    seed_torch(seed=args.seed)


    working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)
    if not os.path.exists(working_directory):
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)


    if not hasattr(args, 'start_time'):
        args.start_time = str(int(datetime.datetime.now().timestamp()))
    if not hasattr(args, 'run_name'):
        args.run_name = f"run-{args.start_time}"

    ## Model Architecture

    base_model = TPN_encoder()

    if args.training_mode != 'none':
        batch_size = args.training_batch_size
        decay_steps = args.training_decay_steps
        epochs = args.training_epochs
        temperature = args.contrastive_temperature
        initial_lr = args.learning_rate

    
    optimizer = torch.optim.Adam(base_model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, last_epoch=-1)

    if args.data_aug == 'none':
        transformation_function = lambda x: x

    dataset = ColloSSL_Dataset(args, transformation_function, all_devices, transfer=False, version=args.version, datasets_name=args.name,)

    train_loader = dataset.get_dataset(split='train')
    val_loader = dataset.get_dataset(split='val')
    
    global all_devices, positive_indices, negative_indices
    all_devices = [args.train_device]
    all_devices.extend(args.positive_devices)
    all_devices.extend(args.negative_devices)
    print("Anchor:", args.train_device, "Positives:", args.positive_devices, "Negatives:", args.negative_devices)
    
    all_devices = list([0, 1, 2, 3, 4])  ## device list
    all_devices.remove(args.train_device)
    all_devices = [args.train_device] + all_devices
        
    anchor_index = 0
    positive_indices = np.arange(len(args.positive_devices)) + 1
    negative_indices = np.arange(len(args.negative_devices)) + 1 + len(args.positive_devices)

    weighted_loss_function = lambda a_e, p_e, p_w, n_e, n_w: weighted_group_contrastive_loss_with_temp(a_e, p_e, p_w, n_e, n_w, temperature=temperature)

    index_mappings = (anchor_index, positive_indices, negative_indices)
    args.index_mappings = index_mappings

    with torch.cuda.device(args.gpu_index):
        collossl = collossl_framework(model=base_model, optimizer=optimizer, scheduler=scheduler, args=args)
        collossl.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
