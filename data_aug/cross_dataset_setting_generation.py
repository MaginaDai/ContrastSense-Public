import numpy as np
import os, pdb
import random
from preprocessing import fetch_instance_number_of_dataset, label_alignment, write_balance_tune_set, write_dataset

datasets = ["HASC", "HHAR", "MotionSense", "Shoaib"]

def merge_samples_from_different_dataset():
    source_root_dir = "datasets/"
    target_dir = "datasets/Merged_dataset"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    sample_id = 0
    for i, dataset in enumerate(datasets):
        total_instance_number = fetch_instance_number_of_dataset(source_root_dir + dataset)
        for j in range(total_instance_number):
            loc = os.path.join(source_root_dir + dataset + '/' + str(j) +'.npz')
            sample = np.load(loc, allow_pickle=True)
            info = sample["add_infor"]
            target_loc = target_dir + '/' + str(sample_id) + '.npz'
            
            if dataset == "HHAR":  # pos3 is previous used on other dataset to denote the cross-user-device/cross-user-position domain.
                info = np.insert(info, 3, -1)
            if dataset == "MotionSense":
                info = np.insert(info, 2, -1)
                info = np.insert(info, 2, -1)
            # pdb.set_trace()
            info = label_alignment(info, dataset)
            if info[0] == -1:
                continue ## we only use samples from classes ['still', 'walk', 'stairsup', 'stairsdown']
            np.savez(target_loc, acc=sample['acc'], gyro=sample['gyro'], add_infor=np.insert(info, 4, i)) # [motions, user_id, device_id, dataset_id, time_infor]
            sample_id += 1
    return

def generate_cross_dataset_setting(seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    dir = "datasets/Merged_dataset/"
    for dataset_idx, dataset in enumerate(datasets):
        target_dir = f"datasets/Merged_dataset_tune_portion_100_{dataset}/"
        segment_merged_dataset(dataset, dataset_idx, dir, target_dir)
    return


def segment_merged_dataset(dataset, dataset_idx, dir, target_dir, ):
    cross = "datasets"
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    domain = []
    motion_label = []
    label_distribution = np.zeros(4)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = np.int32(data['add_infor'][0])
        domain.append(data['add_infor'][4])
        motion_label.append(motion)
        label_distribution[motion] += 1
    
    print(label_distribution)
    # print(f"maximum of domains id {max(domain)}")
    domain_type = np.unique(domain)

    print(f"number of domains {domain_type}")
    
    domains_test_name = [dataset_idx]
    print(f"number of test domains {len(domains_test_name)}")

    domains_train_name = np.delete(domain_type, dataset_idx)
    print(f"number of training domains {len(domains_train_name)}")

    train_all_num = [j for j in range(num) if domain[j] in domains_train_name]
    ## we randomly sample some from the training set/validation set
    np.random.shuffle(train_all_num)
    train_num = train_all_num[:int(len(train_all_num) * 0.85)]
    val_num = train_all_num[int(len(train_all_num) * 0.85):]
    test_num = [j for j in range(num) if domain[j] in domains_test_name]

    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=num, tune_domain_portion=1.0, cross=cross)

if __name__ == "__main__":
    # merge_samples_from_different_dataset()
    
    generate_cross_dataset_setting()