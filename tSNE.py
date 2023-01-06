import torch
import numpy as np
from sklearn.manifold import TSNE
from MoCo import MoCo_model
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.preprocessing import ClassesNum
from figure_plot.figure_plot import t_SNE_view
import matplotlib.pyplot as plt

def load_model(dir, dataset):
    model = MoCo_model(transfer=True, classes=ClassesNum[dataset])
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
        pos = np.where((label_bank[:, 0] == 2) | (label_bank[:, 0] == 5))
        feature_bank = feature_bank[pos]
        label_bank = label_bank[pos, :][0]
        h_trans = tsne.fit_transform(feature_bank)
        motion_type, user_type = np.unique(label_bank[:, 0]), np.unique(label_bank[:, 1])
        plt.figure(figsize=(8, 6))
        
        for i in range(h_trans.shape[0]):
            if label_bank[i, 0] == motion_type[0] and label_bank[i, 1] == user_type[0]:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'ro')
            elif label_bank[i, 0] == motion_type[1] and label_bank[i, 1] == user_type[0]:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'r^')
            elif label_bank[i, 0] == motion_type[0] and label_bank[i, 1] == user_type[1]:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'bo')
            else:
                plt.plot(h_trans[i, 0], h_trans[i, 1], 'b^')
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Feature One", fontsize=14)
        plt.ylabel("Feature Two", fontsize=14)
        
        # t_SNE_view(h_trans, label_bank, version, label_name, dataset_name)
        save_str = f'./figure_plot/tSNE/{dataset_name}_{version}_2user2motion.png'
        plt.savefig(save_str)

    return


if __name__ == '__main__':
    dataset='HHAR'
    dir = f'runs/{dataset}_supervised_plain_no_t/{dataset}_ft_shot_10'
    model_dir = dir + '/model_best.pth.tar'
    version='supervised_plain'
    tSNE_visualization(model_dir, dataset, version)
