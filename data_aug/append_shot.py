import random
import numpy as np
import os
from os.path import dirname
import sys
sys.path.append(dirname(dirname(sys.path[0])))
sys.path.append(dirname(sys.path[0]))
from data_aug.preprocessing import ClassesNum, fetch_instance_number_of_dataset, label_alignment, percent
shot_num = [5, 10, 50, 100, 200] # enlarge to 500

def append_balance_tune_set(ori_dir, target_dir, dataset, dataset_size=None, if_percent=False, if_cross_user=True, tune_domain_portion=0.5, train_dir=None, cross='users'):
    if train_dir is None:
        loc = target_dir + 'train_set' + '.npz'
    else:
        loc = train_dir + 'train_set' + '.npz'
    data = np.load(loc)
    train_set = data['train_set']
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label = []
    domain = []

    for i in train_set:
        sub_dir = ori_dir + i
        data = np.load(sub_dir, allow_pickle=True)
        motion = data['add_infor'][0]
        if cross == 'dataset':
            motion = label_alignment([motion], dataset=dataset)[0]
        label.append(motion)

        if cross == 'users':
            domain.append(data['add_infor'][1])
        elif cross == 'devices':
            domain.append(data['add_infor'][2])
        elif cross == 'positions':
            domain.append(data['add_infor'][2])
        else:
            NotADirectoryError()

    label = np.array(label)
    label_type = np.unique(label)
    label_type_num = len(label_type)
    print(f"type of labels {label_type}")
    if cross == 'dataset':
        assert label_type_num == 4
    else:
        assert label_type_num == ClassesNum[dataset]

    while True:
        domain_type = np.unique(domain)
        ft_shot_dir = train_dir + 'tune_set_0.npz'
        data = np.load(ft_shot_dir)
        tune_set = data['tune_set']
        labeled_domain = []
        for i in tune_set:
            sub_dir = ori_dir + i
            data = np.load(sub_dir, allow_pickle=True)

            if cross == 'users':
                labeled_domain.append(data['add_infor'][1])
            elif cross == 'devices':
                labeled_domain.append(data['add_infor'][2])
            elif cross == 'positions':
                labeled_domain.append(data['add_infor'][2])
            else:
                NotADirectoryError()
        
        domain_selected = np.unique(labeled_domain)

        if dataset_size is None:
            set_size = fetch_instance_number_of_dataset(ori_dir)
        else:
            set_size = dataset_size
        if if_percent:
            for per in percent:
                label_num = per*0.01*set_size
                label_per_class = int(label_num / label_type_num)
                tune_set = []
                counter = []
                if label_per_class < 1:
                    label_per_class = 1  # at least one label for each class
                    print("at least one sample per class")
                
                for i in label_type:
                    idx = np.argwhere(label == i).squeeze()
                    np.random.shuffle(idx)
                    tune_set.extend(train_set[idx[:label_per_class]])
                    counter.append(len(list(idx[:label_per_class])))

                print(f"percent {per}: {len(tune_set)}")
                print(f"each class {counter}")
                loca = target_dir + 'tune_set_' + str(per).replace('.', '_') + '.npz'
                np.savez(loca, tune_set=tune_set)
        else:
            irreasonable_segmentation = 0
            for label_per_class in shot_num:
                tune_set = []
                counter = []
                
                for i in label_type:
                    idx = np.argwhere((label == i)).squeeze()
                    if cross:
                        idx_domain_selected = []
                        for j in idx:
                            if domain[j] in domain_selected:
                                idx_domain_selected.append(j)
                        idx = np.array(idx_domain_selected)
                    np.random.shuffle(idx)
                    if len(idx) == 0:
                        irreasonable_segmentation = 1
                        break
                    if label_per_class == 0:
                        tune_set.extend(train_set[idx])
                        counter.append(len(list(idx)))
                    else:
                        tune_set.extend(train_set[idx[:label_per_class]])
                        counter.append(len(list(idx[:label_per_class])))
                
                if irreasonable_segmentation:
                    break

                print(f"Shot num {label_per_class}: {len(tune_set)}")
                loca = target_dir + 'tune_set_' + str(label_per_class).replace('.', '_') + '.npz'
                np.savez(loca, tune_set=tune_set)
            
            if irreasonable_segmentation == 0:
                break

    print(f"type of domains {domain_type}")
    print(f"motion classes {label_type_num}, total train num {len(label)}, total domain num {len(domain_type)}, domain {domain_selected} provides label")
    return

def append_shot(seed=940, seg_type=5):
    random.seed(seed)
    np.random.seed(seed)
    dataset_name = ["HASC", "HHAR", "Shoaib", "MotionSense"]
    # dataset_name = ["HHAR"]
    for i in range(seg_type):
        for dataset in dataset_name:
            append_balance_tune_set(ori_dir=f'datasets/{dataset}/', train_dir=f'datasets/{dataset}_shot{i}/', target_dir=f'datasets/{dataset}_shot{i}/', dataset=dataset, cross='users')
    return


def append_preliminary(seed=940, seg_type=1):
    dataset="HHAR"
    append_balance_tune_set(ori_dir=f'datasets/{dataset}/', train_dir=f'datasets/HHAR_train65_supervised_label/', target_dir=f'datasets/HHAR_train65_supervised_label/', dataset=dataset, cross='users', tune_domain_portion=0.5)

if __name__ == '__main__':
    # append_shot()
    append_preliminary()