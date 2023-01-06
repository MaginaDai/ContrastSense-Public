import pdb
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib import cm
from scipy.io import loadmat
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import re
import seaborn as sns


def figure_plot_element(y):
    x = np.arange(len(y))
    plt.figure(1)
    plt.xlabel("#")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.legend(["x", "y", "z"])
    plt.show()


def figure_cmp_transformation(y1, y2, name1, name2):
    x = np.arange(len(y1))
    axis_name = ['x', 'y', 'z']
    fig = plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(x, y1[:, i], x, y2[:, i])
        plt.xlabel("Timestamp")
        plt.ylabel(axis_name[i])
        if i == 0:
            plt.title(name1 + " transformed by " + name2)
    fig.legend(labels=['before', 'after'])
    plt.show()



def figure_cmp_models_uci():
    label = [1, 5, 10, 20, 50, 70, 100]
    linear_acc = [58.3, 62.2, 62.5, 63.7, 63.4, 63.9, 63.4]
    fine_tune_acc = [60.1, 61.5, 65.9, 65.5, 66.4, 67.5, 68.7]
    base1_acc = [40.0, 52.3, 58.5, 62.5, 64.2, 66.5, 69.2]
    base2_acc = [22.1, 22.5, 54.4, 59.5, 66.8, 66.2, 71.2]
    plt.figure()
    plt.title("Evaluation on HHAR", fontsize=14)
    plt.plot(label, linear_acc, marker='o')
    plt.plot(label, fine_tune_acc, marker='o')
    plt.plot(label, base1_acc, marker='o', linestyle='dashed')
    plt.plot(label, base2_acc, marker='o', linestyle='dashed')
    plt.xlabel("Percent of Training Instances with Labels", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Linear evaluation', 'Fine-tune', 'Baseline 1', 'Baseline 2'], fontsize=14)
    plt.show()


def figure_cmp_models_uot():
    label = [1, 5, 10, 20, 50, 70, 100]
    fine_tune_acc = [90.8, 95.6, 94.9, 96.1, 97.2, 97.5, 97.5]
    base1_acc = [74.1, 78.7, 89.6, 90.3, 92.5, 93.6, 93.8]
    base2_acc = [55.2, 68.0, 91.9, 93.3, 92.1, 95.4, 95.5]
    plt.figure()
    plt.title("Evaluation on PAR", fontsize=14)
    plt.plot(label, fine_tune_acc, marker='o')
    plt.plot(label, base1_acc, marker='o', linestyle='dashed')
    plt.plot(label, base2_acc, marker='o', linestyle='dashed')
    plt.xlabel("Percent of Training Instances with Labels", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['Fine-tune', 'Baseline 1', 'Baseline 2'], fontsize=14)
    plt.show()


def figure_cmp_acc():
    mine = pd.read_csv('./mine.csv')
    original = pd.read_csv('./paper 8.csv')
    step = mine['Step']
    acc1 = mine['Value']

    acc2 = original['Value']
    plt.figure()
    plt.plot(step, acc1, step, acc2)
    plt.ylabel('Contrastive Accuracy', fontsize=14)
    plt.xlabel('Step', fontsize=14)
    plt.legend(['Ours', 'Transformation in [18]'], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def figure_cmp_loss():
    pattern = r'Loss:\s*(\d\.\d+)'
    epochs = np.arange(100)
    with open('./mine.log', 'r') as fin:
        content = fin.read()
        results1 = re.findall(pattern, content)

    with open('./paper.log', 'r') as fin:
        content = fin.read()
        results2 = re.findall(pattern, content)

    results1 = np.asarray(results1, dtype=float)
    results2 = np.asarray(results2, dtype=float)

    plt.figure()
    plt.plot(epochs, results1, epochs, results2)
    plt.ylabel('Contrastive Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend(['Ours', 'Transformation in [18]'], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def fig_extract_phone_loss():
    pattern = r'accuracy:\s+tensor\(\[*(\d+\.+\d*)\]'
    limu_pattern = r'train\sacc:\s*(-?\d+\.+\d*)'
    loss_pattern = r'Loss:\s*(-?\d+\.+\d+)'
    chs_pattern = r'chs:\s*(\d+\.+\d+)'
    epochs = np.arange(200 -1, step=1)
    with open("./runs/cluster_test_6/training.log", "r") as f:  # 打开文件
        content = f.read()
        u1 = re.findall(pattern, content)
    
    # with open("./baseline/LIMU_BERT/saved/bert_classifier_base_gru_HHAR_20_120/limu_gru_hhar/training.log", "r") as f:  # 打开文件
    #     content = f.read()
    #     u2 = re.findall(limu_pattern, content)

    # with open("./runs/mileStone_g01_w_both/training.log", "r") as f:  # 打开文件
    #     content = f.read()
    #     bl = re.findall(loss_pattern, content)

    u1 = np.asarray(u1, dtype=float)
    # u2 = np.asarray(u2, dtype=float) * 100
    # bl = np.asarray(bl, dtype=float)
    # plt.figure()
    plt.figure(figsize=(7.5, 5.5))
    plt.plot(epochs, u1[epochs], "b")
    # plt.plot(epochs, u2[epochs], "b")
    # plt.plot(epochs, bl[epochs])
    plt.ylabel('Acc', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(["Mine", "LIMU"])
    plt.savefig('Acc_Cluster.png')


def fig_cmp_robustness():
    d = [[90.2, 81.2],
         [89.1, 85.1],
         [89.0, 88.6],
         [87.1, 85.3]]
    d = np.asarray(d, dtype=float)
    
    name_list = ['32','64','128','256']
    
    plt.bar(range(4), d[:, 0], label='All',fc = 'b')
    plt.bar(range(4), d[:, 1], label='Orignal',tick_label = name_list,fc = 'r')

    plt.ylabel('Test Acc', fontsize=14)
    plt.xlabel('Channel Num', fontsize=14)

    plt.legend()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([75, 93])
    plt.savefig('robust.png')


def fig_cluster():
    acc = [48.9, 76.3, 76.0, 75.2]
    x = [6, 60, 250, 500]
    plt.plot(x, acc, '-o')
    plt.xticks([6, 60, 250, 500])
    
    plt.ylabel('Test Acc', fontsize=14)
    plt.xlabel('Clusters Num', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('cluster.png')


def t_SNE_view(x, label, n_step, label_str, dataset):
    label_type = np.unique(label)
    df = np.concatenate((x, np.expand_dims(label, 1)), axis=1)
    df = pd.DataFrame(df, columns=['x', 'y', 'classes'])
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="x", y="y",
        hue="classes",
        palette=sns.color_palette("hls", len(label_type)),
        data=df,
        legend="full",
        alpha=1
    )
    # plt.legend(["User 1", "User 2"])
    save_str = f'./figure_plot/tSNE/{dataset}_{n_step}_test.png'
    plt.savefig(save_str)
    return


def cmp_frequency_performance():
    limu = [97.33, 83.32, 60.41]
    mine = [89.99, 84.41, 70.65]
    x = [20, 50, 100]
    plt.figure()
    plt.plot(x, mine, 'r-o')
    plt.plot(x, limu, 'b-o')
    plt.xticks([20, 50, 100])
    plt.ylabel('Test Acc', fontsize=14)
    plt.xlabel('Frequency', fontsize=14)
    plt.legend(["Mine", "LIMU"])
    plt.savefig(f'./figure_plot/cmp_freq.png')


def cmp_frequency_performance():
    limu = [83.32, 84.81, 83.10]
    mine = [84.41, 83.62, 84.98]
    x = [120, 200, 400]
    plt.figure()
    plt.plot(x, mine, 'r-o')
    plt.plot(x, limu, 'b-o')
    plt.xticks(x)
    plt.ylabel('Test Acc', fontsize=14)
    plt.xlabel('Window Length', fontsize=14)
    plt.legend(["Mine", "LIMU"])
    plt.savefig(f'./figure_plot/cmp_window.png')


def cmp_segmentation_performance():
    limu = [55.34, 62.41, 74.87, 79.19, 84.94]
    mine = [58.03, 73.93, 83.67, 74.60, 83.62]
    x = [1, 2, 3, 4, 5]
    plt.figure()
    plt.plot(x, mine, 'r-o')
    plt.plot(x, limu, 'b-o')
    plt.xticks(x)
    plt.ylabel('Test Acc', fontsize=14)
    plt.xlabel('Users Num for Training', fontsize=14)
    plt.legend(["Mine", "LIMU"])
    plt.savefig(f'./figure_plot/cmp_seg.png')


def figure_supervised_learning():
    y = [34.21, 43.95, 45.9, 53.51]
    x = [1, 5, 15, 50]
    plt.figure()
    plt.plot(x, y, 'r-o')
    plt.xlabel('Shots', fontsize=14)
    plt.ylabel('F1 Score', fontsize=14)
    plt.savefig(f'./figure_plot/DeepSense.png')
    return 


def figure_SCL_during_ft():
    b32_eval = [59.69, 61.19, 60.57, 57.20, 55.54, 58.89, 60.18, 58.81, 56.27, 54.02, 55.65]
    b64_eval = [59.33, 59.66, 60.49, 58.94, 63.02, 59.47, 58.43, 53.53, 55.97, 54.35, 56.17]
    b128_eval = [58.76, 60.50, 56.68, 59.36, 56.30, 54.94, 56.93, 58.87, 55.26, 58.27, 61.43]
    b256_eval = [57.48, 57.62, 57.16, 58.45, 57.61, 57.69, 60.53, ]

    b32_test = [62.06, 60.21, 59.02, 56.14, 55.69, 55.88, 53.86, 58.83, 53.39, 51.32, 52.04]
    b64_test = [60.99, 59.30, 57.15, 59.67, 57.71, 58.36, 52.62, 50.88, 58.01, 54.97, 57.33]
    b128_test = [60.84, 60.54, 60.72, 58.27, 57.84, 57.97, 58.36, 57.79, 58.85, 55.94, 55.60]
    b256_test = [61.65, 60.82, 58.09, 60.31, 61.14, 57.05, 59.01]

    return


def figure_prelim():
    

    return

if __name__ == '__main__':
    # fig_extract_phone_loss()
    # cmp_frequency_performance()
    # cmp_segmentation_performance()
    figure_supervised_learning()
