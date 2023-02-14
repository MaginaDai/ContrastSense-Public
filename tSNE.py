import torch
import numpy as np
from sklearn.manifold import TSNE
from DeepSense import DeepSense_model
from MoCo import MoCo_v1
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, Dataset4Training
from data_aug.preprocessing import ClassesNum
from figure_plot.figure_plot import t_SNE_view
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from data_aug import imu_transforms

def load_model(dir, dataset, model_type, label_type):
    if model_type == 'DeepSense':
        model = DeepSense_model(classes=ClassesNum[dataset])
    elif model_type == 'ContrastSense':
        model = MoCo_v1(device='cpu', label_type=label_type)
    checkpoint = torch.load(dir, map_location="cpu")
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model

label_name = ["walking", "sitting", "standing", "jogging", "biking", "upstairs", "downstairs"]
HHAR_lable = ['stand', 'sit', 'walk', 'stairsup', 'stairsdown', 'bike']

def tSNE_visualization(dir, dataset_name, version):
    model = load_model(dir, dataset_name)
    dataset = ContrastiveLearningDataset(transfer=True, version=version, datasets_name=dataset_name)
    test_dataset = dataset.get_dataset('test')
    feature_bank = torch.empty(0)
    label_bank = torch.empty(0)
    # label_bank_str = []
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    tsne = TSNE(n_components=2, learning_rate='auto', init='random')

    with torch.no_grad():
        for sensor, target in test_loader:
            # feature, _ = model(sensor)
            feature = model.encoder(sensor)
            feature_bank, label_bank = torch.cat((feature_bank, feature)), torch.cat((label_bank, target[:, 0:2]))
        
        feature_bank = feature_bank.reshape(feature_bank.shape[0], -1)
        feature_bank, label_bank = feature_bank.numpy(), label_bank.numpy()
        print(np.unique(label_bank[:, 1]))
        pos = np.where((label_bank[:, 0] == 2) | (label_bank[:, 0] == 5))
        feature_bank = feature_bank[pos]
        label_bank = label_bank[pos, :][0]
        h_trans = tsne.fit_transform(feature_bank)
        motion_type, user_type = np.unique(label_bank[:, 0]), np.unique(label_bank[:, 1])
        plt.figure()
        
        for i in range(h_trans.shape[0]):
            if label_bank[i, 0] == motion_type[0] and label_bank[i, 1] == user_type[0]:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'ro')
            # elif label_bank[i, 0] == motion_type[1] and label_bank[i, 1] == user_type[0]:
            #     plt.plot(h_trans[i, 0], h_trans[i, 1], 'r^')
            elif label_bank[i, 0] == motion_type[0] and label_bank[i, 1] == user_type[1]:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'bo')
            # else:
            #     plt.plot(h_trans[i, 0], h_trans[i, 1], 'b^')
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Feature One", fontsize=16)
        plt.ylabel("Feature Two", fontsize=16)
        plt.tight_layout()
        # t_SNE_view(h_trans, label_bank, version, label_name, dataset_name)
        save_str = f'./figure_plot/tSNE/{dataset_name}_{version}_DeepSense_2user2motion.png'
        plt.savefig(save_str)

    return


def feature_extract(dataset):
    v=0
    version=f"shot{v}"
    dir = f"runs/ablation/CL_v0/{dataset}"
    model_dir = dir + '/model_best.pth.tar'
    CDL_effect_visualize(model_dir, dataset, version, model_name='CL')


    dir = f"runs/CDL_slr0.7_v{v}/{dataset}"
    model_dir = dir + '/model_best.pth.tar'
    CDL_effect_visualize(model_dir, dataset, version, model_name='CDL')


def CDL_effect_visualize(dir, dataset_name, version, model_name):
    if model_name == 'CDL':
        label_type=1
    elif model_name == 'CL':
        label_type=0
    model = load_model(dir, dataset_name, model_type='ContrastSense', label_type=label_type)
    train_dataset = Dataset4Training(dataset_name, version, transform=transforms.Compose([imu_transforms.ToTensor()]), split='train', transfer=True)
    feature_bank = torch.empty(0)
    label_bank = torch.empty(0)
    # label_bank_str = []
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for sensor, target in train_loader:
            # feature, _ = model(sensor)
            feature, _ = model.encoder_q(sensor)
            feature_bank, label_bank = torch.cat((feature_bank, feature)), torch.cat((label_bank, target[:, 0:2]))
        
        feature_bank = feature_bank.reshape(feature_bank.shape[0], -1)
        feature_bank, label_bank = feature_bank.numpy(), label_bank.numpy()
        np.savez(f"{model_name}_{dataset_name}_feature.npz", feature=feature_bank, label=label_bank)


def compare_model_effect(dataset):
    dir_CDL = f"CDL_{dataset}_feature.npz"
    dir_CL = f"CL_{dataset}_feature.npz"
    
    CDL, CL = np.load(dir_CDL), np.load(dir_CL)
    data_cl, label_cl = CL['feature'], CL['label']
    data_cdl, label_cdl = CDL['feature'], CDL['label']
    user = np.unique(label_cl[:, 1])
    user = np.int32(user[[1, 2, 3]])
    
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
        plt.scatter(h_cl[pos, 0], h_cl[pos, 1], color=color_box[i], s=16, label=f"User {i}")
        
    plt.grid(axis='both')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Feature One", fontsize=16)
    # plt.ylabel("Feature Two", fontsize=16)
    plt.title('w/o CDL', fontsize=16)
    
    
    ax2 = plt.subplot(122)
    for i, u in enumerate(user):
        pos = np.int32(np.argwhere(label_cdl[:, 1] == u))
        plt.scatter(h_cdl[pos, 0], h_cdl[pos, 1], color=color_box[i], s=16, label=f"User {i}")
    plt.grid(axis='both')
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel("Feature One", fontsize=16)
    # plt.ylabel("Feature Two", fontsize=16)
    plt.title('w/ CDL', fontsize=16)
    

    fig.legend(['User 0', 'User 1', 'User 2'], fontsize=16,
                    bbox_to_anchor=(0.5, 0.05), loc='center', ncol=3)
    # fig.tight_layout()
    plt.savefig(f"./figure_plot/tSNE/show_CDL_effect_in_{dataset}.pdf", bbox_inches='tight')

color_box = ['#A1A9D0', '#F0988C', '#C4A5DE',  '#F6CAE5', '#96CCCB', '#CFEAF1', '#B883D4', '#9E9E9E', ]

if __name__ == '__main__':
    # dataset='MotionSense'
    # v=0
    # # dir = f'runs/{dataset}_supervised_cross_no_t/{dataset}_ft_shot_10'
    # # dir = f"runs/CDL_slr0.7_v{v}/HHAR"
    # dir = f"runs/ablation/CL_v0/{dataset}"
    # model_dir = dir + '/model_best.pth.tar'
    # # version='supervised_cross'
    # version=f"shot{v}"
    # CDL_effect_visualize(model_dir, dataset, version, model_name='CL')
    dataset='MotionSense'
    # feature_extract(dataset)
    compare_model_effect(dataset)
