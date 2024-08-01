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
from matplotlib import rc
import seaborn as sns
from PIL import Image
# color_blue = '#A1A9D0'
# color_red = '#F0988C'


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
    epochs = np.arange(2000 -1, step=1)
    epochs_2 = np.arange(200 -1, step=1)
    pretrain_file = "runs/slr_weight/hard_v10_cdl_hard_slr0.7_0/HHAR/training.log"
    fine_tune_file = "runs/ewc_results/hard_v10_cdl_hard_ewc50_0/HHAR_ft_shot_10/training.log"
    # with open("./runs/cluster_test_6/training.log", "r") as f:  # 打开文件
    #     content = f.read()
    #     u1 = re.findall(pattern, content)
    
    # with open("./baseline/LIMU_BERT/saved/bert_classifier_base_gru_HHAR_20_120/limu_gru_hhar/training.log", "r") as f:  # 打开文件
    #     content = f.read()
    #     u2 = re.findall(limu_pattern, content)

    with open(pretrain_file, "r") as f:  # 打开文件
        content = f.read()
        bl = re.findall(loss_pattern, content)


    with open(fine_tune_file, "r") as f:  # 打开文件
        content = f.read()
        b2 = re.findall(loss_pattern, content)

    # u1 = np.asarray(u1, dtype=float)
    # u2 = np.asarray(u2, dtype=float) * 100
    bl = np.asarray(bl, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    # plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))
    # plt.subplot(121)
    ax[0].plot(epochs, bl[epochs], "b")
    # plt.plot(epochs, u2[epochs], "b")
    # plt.plot(epochs, bl[epochs])
    ax[0].set_ylabel(r"$L_{pt}$", fontsize=18)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].set_xticks([0, 500, 1000, 1500, 2000], fontsize=16)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].set_title("(a)", fontsize=18, y=-0.4)

    # plt.subplot(122)
    ax[1].plot(epochs_2, b2[epochs_2], "b")
    # plt.plot(epochs, u2[epochs], "b")
    # plt.plot(epochs, bl[epochs])
    ax[1].set_ylabel(r"$L_{ft}$", fontsize=18)
    ax[1].set_xlabel('Epoch', fontsize=16)
    ax[1].set_xticks([0, 50, 100, 150, 200], fontsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].set_title("(b)", fontsize=18, y=-0.4)
    
    plt.subplots_adjust(wspace=0.3)

    fig.savefig('figure_plot/saved/loss.png', bbox_inches='tight')
    fig.savefig('figure_plot/saved/loss.pdf', bbox_inches='tight')


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
    plt.savefig(f'./figure_plot/DeepSense.png',)
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


def figure_domain_shift():
    train_random = [97.83, 94.33, 94.67]
    train_cross = [94.57, 75.18, 58.53]
    # train_label = [88.00, 69.03, 40.79]
    x=np.arange(len(train_random))
    width=0.4
    fig, ax = plt.subplots()
    ax.bar(x-width/2, train_cross, width, color=color_red, label="Setting A")
    ax.bar(x+width/2, train_random, width, color=color_blue, label="Setting B")
    # ax.bar(x+width, train_label, width, color=color_red, label="Setting C")
    ax.set_xticks(x)
    ax.set_xticklabels([r'65', r'45', r'25'])
    # ax.set_xticklabels([r'$\alpha$ = 60', r'$\alpha$ = 25'], fontsize=14)
    ax.legend(fontsize=22, loc='lower right')
    ax.set_ylim(0, 105)
    plt.xlabel(r"Percentage $\alpha$ (%)", fontsize=22)
    plt.ylabel("F1 Score (%)", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.savefig('./figure_plot/preliminary_new.pdf')
    plt.savefig('./figure_plot/preliminary_new.png')
    return

def figure_limited_labels():
    ## on the label source domain setting
    # GILE_result = [43.26, 45.44, 43.49, 43.15, 39.86]
    # CMUDA_result = [70.71, 62.63, 65.02, 45.93, 36.54]
    # FMUDA_result = [73.56, 70.86, 57.80, 46.30, 37.01]
    # supervised_result = [40.79, 37.46, 35.83, 34.23, 24.54]
    # on the cross domain setting
    GILE_result = [63.34, 57.17, 57.81, 42.48, 41.36]
    CMUDA_result = [82.96, 72.66, 69.45, 50.09, 37.86]
    Mixup_result = [71.36, 70.62, 70.14, 67.40, 55.65]
    x=np.arange(len(GILE_result))
    
    plt.figure()
    plt.plot(x, GILE_result, '-o', color=color_box[0], linewidth=2, markersize=10)
    plt.plot(x, Mixup_result, '-v', color=color_red, linewidth=2, markersize=10)
    plt.plot(x, CMUDA_result, '-^', color=color_blue, linewidth=2, markersize=10)
    # plt.plot(x, ewq)
    # plt.plot(x, train_65, 'g-*')
    plt.legend(["GILE", "Mixup", "CMUDA"], fontsize=16)
    plt.xticks(x, labels=["Full", 100, 50, 10, 5])
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r"Shots $n$", fontsize=22)
    plt.ylabel("F1 Score (%)", fontsize=22)
    plt.tight_layout()
    plt.savefig('./figure_plot/limited labels.pdf')
    plt.savefig('./figure_plot/limited labels.png')
    return

def figure_cross_domain(cross):
    Methods = ["LIMU-BERT", "CPCHAR", "FMUDA", "CMUDA", "GILE", "Mixup", "ContrastSense"]
    if cross == "positions":
        results = [39.31, 46.15, 45.28, 47.68, 40.92, 44.21, 60.21]
        std = [15.86, 7.89, 3.36, 5.25, 6.98, 9.23, 5.11]
    elif cross == "devices":
        results = [17.24, 25.52, 20.11, 19.16, 22.67, 22.82, 28.55]
        std = [7.42, 8.0, 2.16, 2.93, 2.07, 1.85, 4.95]
    else:
        NotADirectoryError
    
    x_pos = np.arange(len(Methods))
    fig, ax = plt.subplots()
    ax.bar(x_pos, results, yerr=std, align='center', color=color_blue, ecolor='black', capsize=14)
    ax.set_ylabel('F1-score (%)', fontsize=16)
    ax.set_xticks(x_pos, fontsize=16, bold=True)
    ax.set_xticklabels(Methods, fontsize=10)
    ax.yaxis.grid(True)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(f"figure_plot/{cross}_with_error_bars.pdf")
    # plt.show()

def fig_batch_size_result():
    batch_size = [64, 128, 256, 512]
    x=np.arange(len(batch_size))
    result=[59.52, 59.68, 60.71, 59.15]
    plt.figure()
    plt.bar(x, result, align='center', color=color_blue, capsize=14)
    plt.xticks(x, labels=batch_size, fontsize=16)
    plt.yticks([58, 59, 60, 61], fontsize=16)
    plt.xlabel("Batch Size", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.ylim(58, 61)
    plt.tight_layout()
    plt.savefig('./figure_plot/batch_size_sensativity.png')


def fig_label_domain_portion():
    portion=[40, 60, 80, 100]
    # portion=[33.3, 66.7, 100]
    
    HASC_result = [[25.57, 23.70, 27.46, 29.64],
                   [31.81, 33.04, 37.20, 37.10],
                   [31.75, 31.22, 33.02, 33.41],
                   [35.71, 38.80, 38.42, 36.05],
                   [30.97, 34.40, 35.20, 33.00],
                   [31.86, 40.13, 39.48, 42.69],
                   [42.78, 49.56, 49.25, 51.48]]
    
    HHAR_result = [[55.61, 68.29, 68.45],
                   [61.25, 67.04, 75.91],
                   [41.26, 56.73, 55.50],
                   []]
    
    LIMU_results = [55.14, 56.52, 61.42, 63.00]
    CPC_results = [57.43, 59.43, 65.54, 65.96]
    Mixup_results = [58.83, 59.91, 64.86, 67.74]
    FMUDA_results = [59.53, 58.93, 62.55, 64.76]
    CMUDA_results = [57.75, 59.76, 62.88, 63.63]
    GILE_results = [52.86, 53.04, 58.59, 58.52]
    ContrastSense_results = [65.97, 69.94, 71.18, 72.81]
    Methods = ["LIMU-BERT", "Mixup","GILE", "FMUDA", "CMUDA", "CPCHAR", "ContrastSense"]
    results = np.vstack([LIMU_results, CPC_results, FMUDA_results, CMUDA_results,GILE_results,Mixup_results, ContrastSense_results])
    # print(results)
    x=np.arange(len(portion))

    plt.figure(10)
    for m, method in enumerate(Methods):
        plt.plot(portion, results[m], color=color_box[m], linewidth=3)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Portion of Labeled Source Domains", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.legend(Methods, fontsize=10, loc=4, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('./figure_plot/label_portion_domains_test.png')


def sensitivity_in_one():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    
    queue_size=[256, 512, 1024, 1536, 2048]
    queue_result = [57.64, 58.62, 61.92, 60.46, 60.12]
    err=[2.28, 1.45, 2.88, 2.18, 1.14]
    x=np.arange(len(queue_size))

    ax[0].grid(axis='y')
    ax[0].bar(x, queue_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=2, capthick=2), edgecolor='black', zorder=100)
    ax[0].set_xticks(x, labels=[r"w/o $Q$", "512", "1024", "1536", "2048"], fontsize=22)
    ax[0].set_yticks(np.arange(53, 67, 2), fontsize=22)
    ax[0].tick_params(axis='y', labelsize=22)
    ax[0].set_xlabel(r"(a) The domain queues size $M$", fontsize=22)
    ax[0].set_ylabel("F1 Score (%)", fontsize=22)
    ax[0].set_ylim(53, 67)
    
    
    window=[0, 60, 120, 180, 240]
    window_result = [54.91, 57.34, 57.66, 57.41, 56.19]
    err=[2.23, 1.82, 1.35, 1.06, 2.57]
    x=np.arange(len(window))


    ax[1].grid(axis='y')
    ax[1].bar(x, window_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=2, capthick=2), edgecolor='black', zorder=100)
    ax[1].set_xticks(x, labels=[r"w/o $T$", '60', '120', '180', '240'], fontsize=22)
    ax[1].set_yticks(np.arange(52, 61, 2), fontsize=22)
    ax[1].set_xlabel(r"(b) The time window length $T$", fontsize=22)
    ax[1].set_ylabel("F1 Score (%)", fontsize=22)
    ax[1].set_ylim(52, 61)

    ax[1].tick_params(axis='y', labelsize=22)

    plt.subplots_adjust(wspace=3)
    slr=[0.1, 0.3, 0.5, 0.7, 0.9]
    slr_result = [58.80, 59.04, 58.66, 61.92, 59.74]
    err=[1.31, 2.55, 2.76, 2.88, 2.42]
    x=np.arange(len(slr))
    
    plt.tight_layout()

    plt.savefig(f"./figure_plot/saved/sens.pdf", bbox_inches='tight')
    plt.savefig(f"./figure_plot/saved/sens.png", bbox_inches='tight')

    # plt.savefig('./figure_plot/Queue_size_sensativity.pdf')
    return



def fig_queue_results():
    queue_size=[256, 512, 1024, 1536, 2048]
    queue_result = [57.64, 58.62, 61.92, 60.46, 60.12]
    err=[2.28, 1.45, 2.88, 2.18, 1.14]
    x=np.arange(len(queue_size))
    # plt.subplot(131)
    plt.grid(axis='y')
    plt.bar(x, queue_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=3, capthick=3), edgecolor='black', zorder=100)
    plt.xticks(x, labels=queue_size, fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r"$M$", fontsize=22)
    plt.ylabel("Average F1 Score (%)", fontsize=22)
    plt.ylim(53, 67)
    plt.tight_layout()
    plt.savefig('./figure_plot/Queue_size_sensativity.pdf')
    

def fig_slr_results():
    slr=[0.1, 0.3, 0.5, 0.7, 0.9]
    slr_result = [58.80, 59.04, 58.66, 61.92, 59.74]
    err=[1.31, 2.55, 2.76, 2.88, 2.42]
    x=np.arange(len(slr))
    plt.grid(axis='y')
    plt.bar(x, slr_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=3, capthick=3), edgecolor='black', zorder=100)
    plt.xticks(x, labels=slr, fontsize=22)
    plt.yticks(np.arange(55, 66, 2), fontsize=22)
    plt.xlabel("$\lambda_1$", fontsize=22)
    plt.ylabel("Average F1 Score (%)", fontsize=22)
    plt.ylim(55, 66)
    plt.tight_layout()
    plt.savefig('./figure_plot/Slr_sensativity.png')
    plt.savefig('./figure_plot/Slr_sensativity.pdf')


def fig_time_window_result():
    window=[0, 60, 120, 180, 240]
    window_result = [54.91, 57.34, 57.66, 57.41, 56.19]
    err=[2.23, 1.82, 1.35, 1.06, 2.57]
    x=np.arange(len(window))
    plt.grid(axis='y')
    plt.bar(x, window_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=3, capthick=3), edgecolor='black', zorder=100)
    plt.xticks(x, labels=window, fontsize=22)
    plt.yticks(np.arange(52, 61, 1), fontsize=22)
    plt.xlabel("$T$", fontsize=22)
    plt.ylabel("Average F1 Score (%)", fontsize=22)
    plt.ylim(52, 61)
    plt.tight_layout()
    plt.savefig('./figure_plot/time_window.pdf')
    plt.savefig('./figure_plot/time_window.png')


def fig_negative_sampling_result():
    ratio=[0.4, 0.5, 0.6, 0.7, 0.8]
    HHAR_per=[57.87, 60.79, 59.47, 56.74, 54.89]
    HASC_per=[31.89, 31.31, 30.85, 33.85, 32.98]

    x=np.arange(len(ratio))
    fig, ax1 = plt.subplots()
    
    ax1.plot(x, HHAR_per, '^-', color=color_blue, linewidth=4, markersize=10)
    # ax1.set_xlabel('X-axis', fontsize=22)
    ax1.set_ylabel('HHAR dataset', color=color_blue, fontsize=16)
    ax1.set_xticks(x, labels=ratio, fontsize=16)
    # ax1.set_yticks(np.arange(50, 64, 2), fontsize=22)
    ax1.set_yticks(np.arange(52, 64, 2), )
    ax1.tick_params(axis='y', colors=color_blue, labelsize=16)

    ax2 = ax1.twinx()
    ax2.plot(x, HASC_per, 'o-', color=color_red, linewidth=4, markersize=10)
    ax2.set_ylabel('HASC dataset', color=color_red, fontsize=16)
    ax2.set_yticks(np.arange(29, 38, 2), )
    ax2.tick_params(axis='y', colors=color_red, labelsize=16)

    # plt.show()

    # plt.plot(x, HHAR_per, color=color_blue, linewidth=6,)
    # plt.plot(x, HASC_per, color=color_red, linewidth=6,)
    # plt.xticks(x, labels=ratio, fontsize=22)
    # plt.yticks(np.arange(30, 65, 2), fontsize=22)
    # plt.xlabel("$\lambda_1$", fontsize=22)
    # plt.ylabel("F1 Score (%)", fontsize=22)
    # plt.ylim(30, 65)
    # plt.legend(["HHAR", "HASC"])
    plt.tight_layout()
    plt.savefig('./figure_plot/ratio of similarity sampling.pdf')


def fig_ewc_results():
    # ewc=[0.5, 5, 50, 100, 500]
    # ewc_result = [61.63, 62.06, 62.59, 61.60, 60.83]
    # err=[2.85, 2.63, 2.52, 2.55, 3.13]

    ewc=[1, 25, 50, 75, 100]
    ewc_result=[61.98, 62.17, 62.85, 62.65, 61.17]
    err=[2.34, 2.91, 2.17, 2.50, 1.86]
    x=np.arange(len(ewc))
    plt.grid(axis='y')
    plt.bar(x, ewc_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=3, capthick=3), edgecolor='black', zorder=100)
    plt.xticks(x, labels=ewc, fontsize=22)
    plt.yticks(np.arange(59, 66, 1), fontsize=22)
    plt.xlabel("$\lambda_2$", fontsize=22)
    plt.ylabel("Average F1 Score (%)", fontsize=22)
    plt.ylim(59, 66)
    plt.tight_layout()
    plt.savefig('./figure_plot/EWC_sensativity.pdf')
    plt.savefig('./figure_plot/EWC_sensativity.png')

def fig_time_analysis():
    infoNCE=[1.544, 1.5558, 1.5713, 1.5768, 1.5829]
    RInfo=[1.5809, 1.5936, 1.5950, 1.6061, 1.6052]
    ours=[1.6087, 1.6286, 1.6274, 1.6286, 1.6259]
    q = [256, 512, 1024, 1536, 2048]
    x=np.arange(len(infoNCE))
    width=0.2
    # plt.grid(axis='y')
    plt.bar(x-0.2, infoNCE, color=color_blue, width=width, edgecolor='black', )
    plt.bar(x, RInfo, color=color_box[0], width=width, edgecolor='black', )
    plt.bar(x+0.2, ours, color=color_red, width=width, edgecolor='black', )
    plt.xticks(x, labels=q, fontsize=22)
    plt.yticks(np.arange(1.2, 1.9, 0.2), fontsize=22)
    plt.xlabel("The domain queues size $M$", fontsize=22)
    plt.ylabel("Time (s)", fontsize=22)
    plt.legend(['w/o NegSelet, w/o CDL', 'w/   NegSelet, w/o CDL', 'w/   NegSelet, w/   CDL '], fontsize=16)
    plt.ylim([1.2, 1.9])
    plt.tight_layout()
    plt.savefig('./figure_plot/time_analysis.pdf')
    plt.savefig('./figure_plot/time_analysis.png')

def fig_memory_analysis():
    q = [256, 512, 1024, 1536, 2048]
    # with_queue = [0.2539, 0.5078, 1.0156, 1.5234, 2.0313]
    # without_queue = [1.1758, 2.3516, 4.7031, 7.0547, 9.4063]
    
    with_queue = [3014, 3016, 3018, 3146, 3146]
    without_queue = [2312, 3012, 4222, 6144, 6996]

    x=np.arange(len(q))
    # plt.subplot(132)
    plt.grid(axis='y')
    plt.plot(x, without_queue, '-o', color=color_blue, linewidth=2, markersize=10)
    plt.plot(x, with_queue, '-^', color=color_red, linewidth=2, markersize=10)
    plt.legend([r'w/o $Q$', r'w/ $Q$'], fontsize=16)
    plt.xticks(x, labels=q, fontsize=22)
    plt.yticks(np.arange(1000, 7000, 1000), fontsize=22)
    plt.xlabel("The number of negatives", fontsize=22)
    plt.ylabel("Memory (MB)", fontsize=22)
    # plt.title('(b)', fontsize=16, y=-0.4)
    # plt.ylim(57, 63)
    plt.tight_layout()
    plt.savefig('./figure_plot/memory_analysis.pdf')

def fig_cost_in_one_figure():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))

    infoNCE=[1.544, 1.5558, 1.5713, 1.5768, 1.5829]
    RInfo=[1.5809, 1.5936, 1.5950, 1.6061, 1.6052]
    ours=[1.6087, 1.6286, 1.6274, 1.6286, 1.6259]
    q = [256, 512, 1024, 1536, 2048]
    x=np.arange(len(infoNCE))
    width=0.2
    # plt.grid(axis='y')
    
    ax[0].bar(x-0.2, infoNCE, color=color_blue, width=width, edgecolor='black', )
    ax[0].bar(x, RInfo, color=color_box[0], width=width, edgecolor='black', )
    ax[0].bar(x+0.2, ours, color=color_red, width=width, edgecolor='black', )
    ax[0].set_xticks(x, labels=["w/o Q", "512", "1024", "1536", "2048"], fontsize=14)
    ax[0].set_ylim([1.3, 2.0])
    ax[0].set_yticks(np.arange(1.3, 2.0, 0.2), fontsize=20)
    ax[0].set_xlabel("The domain queues size $M$", fontsize=20)
    ax[0].set_ylabel("Time (s)", fontsize=20)
    ax[0].legend(['w/o NegSelet, w/o CDL', 'w/   NegSelet, w/o CDL', 'w/   NegSelet, w/   CDL '], fontsize=16, loc='upper right')
    
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].set_title("(a) Time consumption", fontsize=20, y=-0.4)


    with_queue = [3014, 3016, 3018, 3146, 3146]
    without_queue = [2312, 3012, 4222, 6144, 6996]

    x=np.arange(len(q))
    # plt.subplot(132)
    ax[1].grid(axis='y')
    ax[1].plot(x, without_queue, '-o', color=color_blue, linewidth=2, markersize=10)
    ax[1].plot(x, with_queue, '-^', color=color_red, linewidth=2, markersize=10)
    ax[1].legend([r'w/o $Q$', r'w/ $Q$'], fontsize=16)
    ax[1].set_xticks(x, labels=q, fontsize=20)
    ax[1].set_yticks(np.arange(1000, 7000, 1000), fontsize=20)
    ax[1].set_xlabel("The number of negatives", fontsize=20)
    ax[1].set_ylabel("Memory (MB)", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=20)
    ax[1].set_title("(b) Memory usage", fontsize=20, y=-0.4)

    plt.subplots_adjust(wspace=3)

    plt.tight_layout()
    plt.savefig('./figure_plot/saved/cost_in_one.pdf', bbox_inches='tight')
    plt.savefig('./figure_plot/saved/cost_in_one.png', bbox_inches='tight')

    return

def fig_aug_effect():
    name = ['All', 'w/o Negate', 'w/o Scale', 'w/o Wrap', 'w/o Flip', 'w/o Noise', 'w/o Rotate']
    results = [62.59, 60.37, 59.94, 59.65, 60.97, 60.04, 57.73]
    std_results = [2.52, 2.87, 2.41, 2.45, 3.31, 2.66, 1.52]
    x=np.arange(len(name))
    plt.figure()
    # ax = plt.figure(figsize=(6, 4))
    plt.grid(axis='y', zorder=0)
    plt.bar(x, results, align='center', color=color_blue, capsize=10, yerr=std_results, ecolor='black', error_kw=dict(elinewidth=3, capthick=3), edgecolor='black', zorder=100)
    plt.xticks(x, labels=name, fontsize=22, rotation=30)
    plt.yticks(np.arange(50, 66, 2), fontsize=22)
    plt.ylabel("F1 Score (%)", fontsize=22)
    plt.ylim(50, 66)
    plt.tight_layout()
    plt.savefig('./figure_plot/aug_effect.pdf')

def fig_sensativity_analysis():
    plt.figure(figsize=(16, 3))
    queue_size=[256, 512, 1024, 2048]
    queue_result = [59.49, 60.38, 62.59, 61.29]
    x=np.arange(len(queue_size))
    plt.subplot(131)
    plt.grid(axis='y')
    plt.bar(x, queue_result, align='center', color=color_blue, capsize=14)
    plt.xticks(x, labels=queue_size, fontsize=16)
    plt.yticks(np.arange(57, 64), fontsize=16)
    plt.xlabel("Domain Queue Size", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.title('(a)', fontsize=16, y=-0.4)
    plt.ylim(56, 63)
    
    slr=[0.1, 0.3, 0.5, 0.7, 0.9]
    slr_result = [59.75, 60.71, 60.50, 62.59, 60.16]
    x=np.arange(len(slr))
    plt.subplot(132)
    plt.grid(axis='y')
    plt.plot(x, slr_result, '-^', color=color_blue, linewidth=3)
    plt.xticks(x, labels=slr, fontsize=16)
    plt.yticks(np.arange(57, 64), fontsize=16)
    plt.xlabel("$\lambda_1$", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.title('(b)', fontsize=16, y=-0.4)
    plt.ylim(57, 63)

    ewc=[0.5, 5, 50, 100, 500]
    ewc_result = [61.63, 62.06, 62.59, 61.60, 60.83]
    x=np.arange(len(ewc))
    plt.subplot(133)
    plt.grid(axis='y')
    plt.plot(x, ewc_result, '-^', color=color_blue, linewidth=3)
    plt.xticks(x, labels=ewc, fontsize=16)
    plt.yticks(np.arange(59, 64), fontsize=16)
    plt.xlabel("$\lambda_2$", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.ylim(59, 63)
    plt.title('(c)', fontsize=16, y=-0.4)
    plt.savefig('./figure_plot/sensitivity_analysis.png', bbox_inches='tight')
    return

def figure_domain_shift_new_previous():
    cpc_random = [94.76, 95.62, 95.87]
    cpc_label = [94.91, 74.68, 51.61]
    
    limu_random = [95.62, 96.66, 95.11]
    limu_label = [95.90, 78.70, 56.87]
    
    x=np.arange(len(cpc_random)*2)
    width=0.4
    interval=0.4

    fig, ax = plt.subplots()
    # print(fig.size())
    ax.bar(x[0:3]-interval-width/2, cpc_label, width, color=color_red, label="Setting A")
    ax.bar(x[0:3]-interval+width/2, cpc_random, width, color=color_blue, label="Setting B")

    ax.bar(x[3:]+interval-width/2, limu_label, width, color=color_red)
    ax.bar(x[3:]+interval+width/2, limu_random, width, color=color_blue)
    ax.axvline((x[2]+x[3])/2, linestyle='--', color="#b4b4b4")

    # ax.plot(x[0:3]-interval-width/2, cpc_random, '--', color='black', linewidth=3)
    # ax.plot(x[0:3]-interval+width/2, cpc_label, '--', color='black', linewidth=2)

    # ax.plot(x[3:]+interval-width/2, limu_random, '--', color='black', linewidth=3)
    # ax.plot(x[3:]+interval+width/2, limu_label, '--', color='black', linewidth=2)

    
    # ax.bar(x+width, train_label, width, color=color_red, label="Setting C")
    ax.set_xticks(np.concatenate([x[0:3]-interval, x[3:]+interval]))
    ax.set_xticklabels([r"65", r"45", r"25", r"65", r"45", r"25"])
    # ax.set_xticklabels([r'$\alpha$ = 60', r'$\alpha$ = 25'], fontsize=14)
    ax.legend(fontsize=14, loc='lower right', ncol=1)
    ax.set_ylim(40, 100)
    # plt.xlabel(r"Percentage $\alpha$ (%)", fontsize=16)
    plt.xlabel("CPCHAR                           LIMU", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('./figure_plot/preliminary_sparse.png')

def figure_domain_shift_new():
    cpc_random = [94.76, 95.62, 95.87]
    cpc_label = [94.91, 74.68, 51.61]
    
    limu_random = [95.62, 96.66, 95.11]
    limu_label = [95.90, 78.70, 56.87]

    x=np.arange(len(cpc_random))
    width=0.3

    fig, ax = plt.subplots(2)
    # print(fig.size())
    ax[0].bar(x-width/2, cpc_label, width, color=color_red, label="Setting A", edgecolor='black')
    ax[0].bar(x+width/2, cpc_random, width, color=color_blue, label="Setting B", edgecolor='black')
    ax[0].set_ylim(40, 100)
    ax[0].set_xlim(-0.6, 2.6)
    # plt.xlabel(r"Percentage $\alpha$", fontsize=16)
    ax[0].set_ylabel("F1 Score (%)", fontsize=20)
    ax[0].set_xlabel("CPCHAR", fontsize=22)
    ax[0].set_xticks(x, labels=[r"$\alpha=65\%$", r"$\alpha=45\%$", r"$\alpha=25\%$"], fontsize=22)
    ax[0].set_yticks(np.arange(40,120,20), fontsize=22)
    ax[0].tick_params(axis='y', labelsize=22)
    ax[0].legend(fontsize=16, loc='lower left', ncol=1)
   

    ax[1].bar(x-width/2, limu_label, width, color=color_red, label="Setting A", edgecolor='black')
    ax[1].bar(x+width/2, limu_random, width, color=color_blue, label="Setting B", edgecolor='black')
    # ax.axvline((x[2]+x[3])/2, linestyle='--', color="#b4b4b4")
    ax[1].set_ylim(40, 100)
    ax[1].set_xlim(-0.6, 2.6)
    

    # plt.xlabel(r"Percentage $\alpha$", fontsize=16)
    # plt.ylabel("F1 Score (%)", fontsize=16)
    ax[1].tick_params(axis='y', labelsize=22)
    ax[1].set_ylabel("F1 Score (%)", fontsize=20)
    ax[1].set_xticks(x, labels=[r"$\alpha=65\%$", r"$\alpha=45\%$", r"$\alpha=25\%$"], fontsize=22)
    ax[1].set_yticks(np.arange(40,120,20), fontsize=22)
    ax[1].set_xlabel("LIMU-BERT", fontsize=22)
    ax[1].legend(fontsize=16, loc='lower left', ncol=1)
    plt.tight_layout()
    plt.savefig('./figure_plot/preliminary_new.pdf')
    plt.savefig('./figure_plot/preliminary_new.png')


def add_axis_to_tSNE():
    path='figure_plot/tSNE/show_CDL_effect_in_MotionSense_t.png'

    image = Image.open(path)

    fig, ax = plt.subplots()

    # Display the image using imshow
    ax.imshow(image)

    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])

    fig.savefig('figure_plot/tSNE/show_CDL_effect_in_MotionSense_t_with_axis.png')


def Momemtum_update_rate_plot():
    rate=[0, 0.9, 0.99, 0.999, 1.0]
    f1=[49.08, 60.96, 62.45, 62.85, 56.39]
    x=np.arange(len(rate))
    # plt.subplot(131)
    plt.grid(axis='y')
    plt.bar(x, f1, align='center', color=color_blue, capsize=14)
    plt.xticks(x, labels=rate, fontsize=16)
    plt.yticks(np.arange(45, 65, 2), fontsize=16)
    plt.xlabel("Momemtum Update Rate", fontsize=16)
    plt.ylabel("F1 Score (%)", fontsize=16)
    plt.title('(a)', fontsize=16, y=-0.25)
    plt.ylim(45, 65)
    plt.tight_layout()
    plt.savefig('./figure_plot/momentum_update_rate.png')
    

def effectiveness_of_domain_queues_in_one():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    
    queue_size=[256, 512, 1024, 1536, 2048]
    queue_result = [57.64, 58.62, 61.92, 60.46, 60.12]
    err=[2.28, 1.45, 2.88, 2.18, 1.14]
    x=np.arange(len(queue_size))

    ax[0].grid(axis='y')
    ax[0].bar(x, queue_result, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=2, capthick=2), edgecolor='black', zorder=100)
    ax[0].set_xticks(x, labels=[r"w/o $Q$", "512", "1024", "1536", "2048"], fontsize=20)
    ax[0].set_yticks(np.arange(53, 67, 2), fontsize=22)
    ax[0].tick_params(axis='y', labelsize=22)
    ax[0].set_xlabel("The domain queues size M", fontsize=22)
    ax[0].set_ylabel("Average F1 Score (%)", fontsize=19)
    ax[0].set_ylim(53, 67)
    ax[0].set_title("(a)", fontsize=20, y=-0.4)

    
    rate=[0.8, 0.9, 0.99, 0.999, 1.0]
    f1=[53.98, 60.96, 62.45, 62.85, 56.39]
    err=[5.70, 4.11, 1.62, 2.88, 1.28]
    x=np.arange(len(rate))


    ax[2].grid(axis='y')
    ax[2].bar(x, f1, align='center', color=color_blue, yerr=err, capsize=10, ecolor='black', error_kw=dict(elinewidth=2, capthick=2), edgecolor='black', zorder=100)
    ax[2].set_xticks(x, labels=["0.8", '0.9', '0.99', '0.999', '1.0'], fontsize=20)
    ax[2].set_yticks(np.arange(46, 68, 4), fontsize=22)
    ax[2].set_xlabel("The Momentum update ratio", fontsize=22)
    ax[2].set_ylabel("Average F1 Score (%)", fontsize=19)
    ax[2].set_ylim(46, 68)
    ax[2].set_title("(c)", fontsize=20, y=-0.4)

    ax[2].tick_params(axis='y', labelsize=22)

    plt.subplots_adjust(wspace=3)
    slr=[0.1, 0.3, 0.5, 0.7, 0.9]
    slr_result = [58.80, 59.04, 58.66, 61.92, 59.74]
    err=[1.31, 2.55, 2.76, 2.88, 2.42]
    x=np.arange(len(slr))

    with_queue = [3014, 3016, 3018, 3146, 3146]
    without_queue = [2312, 3012, 4222, 6144, 6996]
    q = [256, 512, 1024, 1536, 2048]
    x=np.arange(len(q))
    # plt.subplot(132)
    ax[1].grid(axis='y')
    ax[1].plot(x, without_queue, '-o', color=color_blue, linewidth=2, markersize=10)
    ax[1].plot(x, with_queue, '-^', color=color_red, linewidth=2, markersize=10)
    ax[1].legend([r'w/o $Q$', r'w/ $Q$'], fontsize=16)
    ax[1].set_xticks(x, labels=q, fontsize=20)
    ax[1].set_yticks(np.arange(1000, 7000, 1000), fontsize=20)
    ax[1].set_xlabel("The number of negatives", fontsize=20)
    ax[1].set_ylabel("Memory (MB)", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=20)
    ax[1].set_title("(b)", fontsize=20, y=-0.4)
    
    plt.tight_layout()

    plt.savefig(f"./figure_plot/saved/dq.pdf", bbox_inches='tight')
    plt.savefig(f"./figure_plot/saved/dq.png", bbox_inches='tight')


def fig_memory_f1_tradeoff():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(0.8, 1.6, 0.2)
    memory_space = [0.83, 0.84, 0.93, 0.95, 0.83, 0.84, 0.90]
    f1_score = [46.38, 44.87, 53.49, 46.29, 53.93, 50.76, 51.57]
    
    memory_CS = [0.82, 0.98, 1.48]
    ContrastSense = [59.42, 60.86, 62.85]
    ax[0].plot(memory_space, f1_score, 'o', color=color_blue, markersize=10)
    ax[0].plot(memory_CS, ContrastSense, '--^', color=color_red, markersize=10)
    ax[0].set_xticks(x, labels = np.arange(0.8, 1.6, 0.2), fontsize=22)
    ax[0].set_yticks(np.arange(40, 67, 4), fontsize=22)
    ax[0].tick_params(axis='y', labelsize=22)
    ax[0].set_xlabel("(a) The memory usage (%)", fontsize=22)
    ax[0].set_ylabel("F1 Score (%)", fontsize=22)
    ax[0].set_ylim(40, 67)
    # ax[0].set_title("", fontsize=20, y=-0.4)
    plt.tight_layout()

    plt.savefig(f"./figure_plot/saved/memory_f1.pdf", bbox_inches='tight')
    plt.savefig(f"./figure_plot/saved/memory_f1.png", bbox_inches='tight')
    return


color_blue = '#3c75b0'
color_red = '#e6843b'
# color_box = ['#A1A9D0', '#96CCCB', '#B883D4', '#9E9E9E', '#CFEAF1', '#C4A5DE', '#F0988C', '#F6CAE5']
color_box = ['#529e52', '#e6843b', '#c53a32', '#529e52', '#e6843b', '#c53a32', '#3c75b0']

if __name__ == '__main__':
    # fig_extract_phone_loss()
    # cmp_frequency_performance()
    # cmp_segmentation_performance()
    # figure_supervised_learning()
    # figure_domain_shift()
    # figure_limited_labels()
    # Momemtum_update_rate_plot()
    # figure_cross_domain(cross='positions')
    # fig_label_domain_portion()
    # fig_batch_size_result()
    # fig_aug_effect()
    # fig_queue_results()
    # fig_time_window_result()
    # fig_slr_results()
    # fig_ewc_results()
    # fig_sensativity_analysis()
    # figure_domain_shift_new()
    # fig_time_analysis()
    # fig_memory_analysis()
    # add_axis_to_tSNE()

    # fig_negative_sampling_result()
    # fig_time_window_result()

    # sensitivity_in_one()

    # fig_cost_in_one_figure()
    effectiveness_of_domain_queues_in_one()
    # fig_memory_f1_tradeoff()