import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
from DeepSense import DeepSense_model
from MoCo import MoCo_model, MoCo_v1
from baseline.CPCHAR.CPC import Transfer_Coder
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_aug.preprocessing import ClassesNum
from figure_plot.figure_plot import t_SNE_view
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from data_aug import imu_transforms

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D

def updateline(handle, orig):
    handle.update_from(orig)
    handle.set_markersize(30)

def load_model(dir, dataset, model_type, label_type=0):
    if model_type == 'DeepSense':
        model = DeepSense_model(classes=ClassesNum[dataset])
    elif model_type == 'ContrastSense':
        model = MoCo_v1(device='cpu', label_type=label_type)
    elif model_type == 'ContrastSense_ft':
        model = MoCo_model(transfer=True, classes=ClassesNum[dataset])
    elif model_type == 'CPC':
        parser = argparse.ArgumentParser(description='PyTorch Contrastive Learning for Wearable Sensing')
        args = parser.parse_args()
        args.kernel_size = 3
        args.padding = int(args.kernel_size // 2)
        args.input_size = 6
        model = Transfer_Coder(classes=ClassesNum[dataset], args=args)
    checkpoint = torch.load(dir, map_location="cpu")
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model

label_name = ["walking", "sitting", "standing", "jogging", "biking", "upstairs", "downstairs"]
HHAR_lable = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']

def tSNE_visualization(dir, dataset_name, version, model_type='CPC', shot=10, gpu_idx=2):
    device = torch.device(f'cuda:{gpu_idx}')
    model = load_model(dir, dataset_name, model_type=model_type)
    model = model.to(device)
    dataset = ContrastiveLearningDataset(transfer=True, version=version, datasets_name=dataset_name)
    
    train_dataset = Dataset4Training(dataset_name, version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='train', transfer=True)
    tune_dataset = Dataset4Training(dataset_name, version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='tune', transfer=True, shot=shot)
    test_dataset = dataset.get_dataset('test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, pin_memory=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=False, drop_last=False)
    
    feature_bank = torch.empty(0).to(device)
    label_bank = torch.empty(0)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    
    train_user = []
    tune_user = []
    test_user = []
    for sensor, target in train_dataset:
        train_user.append(target[1].numpy())
    for sensor, target in tune_dataset:
        tune_user.append(target[1].numpy())
    for sensor, target in test_dataset:
        test_user.append(target[1].numpy())
    
    train_user = np.unique(train_user)
    tune_user = np.unique(tune_user)
    test_user = np.unique(test_user)
    # select_user = [train_user[1], tune_user[0], test_user[0]]
    print(train_user)
    print(tune_user)
    print(test_user)

    select_user = [6, 5, 2]

    with torch.no_grad():
        for sensor, target in tqdm(train_loader):
            sensor = sensor.to(device)
            if model_type == 'CPC':
                sensor = sensor.squeeze(1)
                feature = model.ar(model.encoder(sensor))
            else:
                sensor = sensor.reshape(-1, sensor.shape[0], sensor.shape[1], sensor.shape[2])
                feature = model.encoder(sensor)
            feature = feature.reshape(sensor.shape[0], -1)
            pos = [i for i, t in enumerate(target) if t[1] in select_user]
            feature_bank, label_bank = torch.cat((feature_bank, feature[pos])), torch.cat((label_bank, target[pos, 1]))

        for sensor, target in tqdm(test_loader):
            sensor = sensor.to(device)
            if model_type == 'CPC':
                sensor = sensor.squeeze(1)
                feature = model.ar(model.encoder(sensor))
            else:
                sensor = sensor.reshape(-1, sensor.shape[0], sensor.shape[1], sensor.shape[2])
                feature = model.encoder(sensor)
            feature = feature.reshape(sensor.shape[0], -1)
            pos = [i for i, t in enumerate(target) if t[1] in select_user]
            
            feature_bank, label_bank = torch.cat((feature_bank, feature[pos])), torch.cat((label_bank, target[pos, 1]))
        
        feature_bank = feature_bank.reshape(feature_bank.shape[0], -1)
        feature_bank, label_bank = feature_bank.cpu().numpy(), label_bank.numpy()

        h_trans = tsne.fit_transform(feature_bank)

    plt.figure()

    for i, u in enumerate(select_user):
        pos = np.int32(np.argwhere(label_bank == u)).reshape(-1)
        select_pos_len = int(len(pos)*0.2)
        select_pos = np.random.choice(pos, size=select_pos_len)

        # if i == 2:
        plt.scatter(h_trans[select_pos, 0], h_trans[select_pos, 1], color=color_box[i], s=20, label=f"User {i}", marker=marker_box[i])
        # else:
        #     plt.scatter(h_trans[pos, 0], h_trans[pos, 1], edgecolors=color_box[i], s=30, label=f"User {i}", marker=marker_box[i], c='white')

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Feature One", fontsize=16)
    plt.ylabel("Feature Two", fontsize=16)
    legd = plt.legend(['User 1', 'User 2', 'User 3'], fontsize=12)
    legd.legendHandles[0].set_sizes([60])
    legd.legendHandles[1].set_sizes([60])
    legd.legendHandles[2].set_sizes([60])

    save_str = f'./figure_plot/tSNE/{dataset_name}_{version}_{model_type}_2user2motion_t.pdf'
    save_str_png = f'./figure_plot/tSNE/{dataset_name}_{version}_{model_type}_2user2motion_t.png'
    plt.savefig(save_str, bbox_inches='tight')
    plt.savefig(save_str_png, bbox_inches='tight')
    return


def feature_extract(dataset, v):
    version=f"shot{v}"
    # dir = f"runs/ablation/CL_v{v}/{dataset}"
    dir = f"runs/ablation/CL_no_ft_design_v{v}/{dataset}_ft_shot_10"
    model_dir = dir + '/model_best.pth.tar'
    CDL_effect_visualize(model_dir, dataset, version, model_name='CL', model_type='ContrastSense_ft')


    # dir = f"runs/CDL_slr0.7_v{v}/{dataset}"
    dir = f"runs/CDL_shot_num/CDL_shot_number_v{v}/{dataset}_ft_shot_10"
    model_dir = dir + '/model_best.pth.tar'
    CDL_effect_visualize(model_dir, dataset, version, model_name='CDL', model_type='ContrastSense_ft')


def CDL_effect_visualize(dir, dataset_name, version, model_name, model_type):
    if model_name == 'CDL':
        label_type=1
    elif model_name == 'CL':
        label_type=0
    model = load_model(dir, dataset_name, model_type=model_type, label_type=label_type)
    train_dataset = Dataset4Training(dataset_name, version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='train', transfer=True)
    feature_bank = torch.empty(0)
    label_bank = torch.empty(0)
    # label_bank_str = []
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for sensor, target in train_loader:
            # feature, _ = model(sensor)
            # feature, _ = model.encoder_q(sensor)
            feature = model.encoder(sensor)
            feature_bank, label_bank = torch.cat((feature_bank, feature)), torch.cat((label_bank, target[:, 0:2]))
        
        feature_bank = feature_bank.reshape(feature_bank.shape[0], -1)
        feature_bank, label_bank = feature_bank.numpy(), label_bank.numpy()
        np.savez(f"{model_name}_{dataset_name}_feature.npz", feature=feature_bank, label=label_bank)


def compare_model_effect(dataset):
    dir_CDL = f"figure_plot/CDL_{dataset}_feature.npz"
    dir_CL = f"figure_plot/CL_{dataset}_feature.npz"

    # dir_CDL = f"CDL_{dataset}_feature.npz"
    # dir_CL = f"CL_{dataset}_feature.npz"
    
    CDL, CL = np.load(dir_CDL), np.load(dir_CL)
    data_cl, label_cl = CL['feature'], CL['label']
    data_cdl, label_cdl = CDL['feature'], CDL['label']
    user = np.unique(label_cl[:, 1])
    user = np.int32(user[[1, 2, 3]])

    # user = np.int32(user[:])
    
    pos=[]
    for u in user:
        pos = np.concatenate((pos, np.where(label_cl[:, 1] == u)[0]))
    pos=np.int32(np.sort(pos))
    data_cl = data_cl[pos]
    label_cl = label_cl[pos]

    pos=[]
    for u in user:
        pos = np.concatenate((pos, np.where(label_cdl[:, 1] == u)[0]))
    pos=np.int32(np.sort(pos))
    data_cdl = data_cdl[pos]
    label_cdl = label_cdl[pos]

    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    
    h_cl = tsne.fit_transform(data_cl)
    h_cdl = tsne.fit_transform(data_cdl)
    
    # plt.figure((10, 6))
    fig = plt.figure(figsize=(10, 4))
    
    ax1 = plt.subplot(121)
    for i, u in enumerate(user):
        pos = np.int32(np.argwhere(label_cl[:, 1] == u))
        plt.scatter(h_cl[pos, 0], h_cl[pos, 1], color=color_box[i], s=30, label=f"User {i}", marker=marker_box[i])
        
    plt.grid(axis='both')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Feature One", fontsize=16)
    # plt.ylabel("Feature Two", fontsize=16)
    plt.title('(a) w/o CDL', fontsize=16, y=-0.15)
    
    ax2 = plt.subplot(122)
    for i, u in enumerate(user):
        pos = np.int32(np.argwhere(label_cdl[:, 1] == u))
        plt.scatter(h_cdl[pos, 0], h_cdl[pos, 1], color=color_box[i], s=30, label=f"User {i}", marker=marker_box[i])
    plt.grid(axis='both')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Feature One", fontsize=16)
    # plt.ylabel("Feature Two", fontsize=16)
    plt.title('(b) w/ CDL', fontsize=16, y=-0.15)
    
    legd = fig.legend(['User 0', 'User 1', 'User 2'], fontsize=16, 
                bbox_to_anchor=(0.5, 1.00), loc='center', ncol=3,)
    legd.legendHandles[0].set_sizes([60])
    legd.legendHandles[1].set_sizes([60])
    legd.legendHandles[2].set_sizes([60])
    # fig.tight_layout()
    plt.savefig(f"./figure_plot/tSNE/show_CDL_effect_in_{dataset}_t.png", bbox_inches='tight')
    plt.savefig(f"./figure_plot/tSNE/show_CDL_effect_in_{dataset}_t.pdf", bbox_inches='tight')

# color_box = ['#A1A9D0', '#F0988C', '#C4A5DE',  '#F6CAE5', '#96CCCB', '#CFEAF1', '#B883D4', '#9E9E9E', ]
color_box = ['#529e52', '#e6843b', '#3c75b0', '#c53a32', '#529e52', '#e6843b', '#c53a32',  ]
marker_box=['o', '^', 'v']

if __name__ == '__main__':
    v=0
    # dataset='HHAR'
    # dir = f"baseline/CPCHAR/runs/CPCHAR_cu{v}/{dataset}_ft_shot_50"
    # model_dir = dir + '/model_best.pth.tar'
    # version=f"shot{v}"
    # dataset='MotionSense'
    # feature_extract(dataset, v=2)
    # compare_model_effect(dataset)

    dir = f"baseline/CPCHAR/runs/25_label/HHAR_ft_shot_50"
    model_dir = dir + '/model_best.pth.tar'
    version=f"domain_shift"
    dataset='HHAR'
    tSNE_visualization(model_dir, dataset, version, model_type='CPC', shot=0, gpu_idx=1)

    
