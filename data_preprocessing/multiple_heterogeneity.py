
import numpy as np
from data_preprocessing.data_split import fetch_instance_number_of_dataset, write_balance_tune_set, write_dataset
import random, os

def preprocessing_dataset_cross_multiple_domains(dir, target_dir, dataset, test_portion=0.60, val_portion=0.15, tune_domain_portion=0.40, cross='users_devices'):
    print(dataset)
    
    num = fetch_instance_number_of_dataset(dir)
    domain = []
    motion_label = []
    label_distribution = np.zeros(7)
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        data = np.load(sub_dir, allow_pickle=True)
        motion = np.int32(data['add_infor'][0])
        domain.append([np.float64(data['add_infor'][1]), np.float64(data['add_infor'][2])])  # jointly decide the domains
        motion_label.append(motion)
        label_distribution[motion] += 1
    
    print(label_distribution)
    # print(f"maximum of domains id {max(domain)}")
    domain_type = np.unique(domain, axis=0)
    test_num = max(int(len(domain_type) * test_portion), 1)
    val_num = max(int(len(domain_type) * val_portion), 1)
    
    print(f"number of domains {domain_type}")
    np.random.shuffle(domain_type)
    domains_test_name = domain_type[:test_num]
    # users_test_name = np.array(['e', 'i'])
    print(f"number of test domains {len(domains_test_name)}")

    domains_train_name = domain_type[test_num+val_num:]
    # users_train_name = np.array(['a', 'd', 'f', 'g', 'h'])
    print(f"number of training domains {len(domains_train_name)}")
    
    domains_val_name = domain_type[test_num:test_num+val_num]
    # users_val_name = np.array(['b', 'c'])
    print(f"number of validation domains {len(domains_val_name)}")

    train_num=[]
    val_num=[]
    test_num=[]

    for j in range(num):
        if np.any([np.all(domain[j] == domains_train) for domains_train in domains_train_name]):
            train_num.append(j)
        elif np.any([np.all(domain[j] == domains_val) for domains_val in domains_val_name]):
            val_num.append(j)
        elif np.any([np.all(domain[j] == domains_test) for domains_test in domains_test_name]):
            test_num.append(j)
        else:
            print("now")


    write_dataset(target_dir, train_num, val_num, test_num)
    write_balance_tune_set(dir, target_dir, dataset, dataset_size=num, tune_domain_portion=tune_domain_portion, cross=cross)
    return


def new_segmentation_for_multiple_domain_shift(seg_types=5, seed=940):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # dataset_name = ["HASC"]
    dataset_name = ["Shoaib"]
    for i in range(seg_types):
        for dataset in dataset_name:
            preprocessing_dataset_cross_multiple_domains(dir=f'datasets/{dataset}/', target_dir=f"datasets/{dataset}_users_positions_alpha45_shot{i}/", dataset=dataset, cross='multiple', test_portion=0.4)

    return


def new_tune_segmentation_cross_multiple_domains_with_different_portion(seed=940, seg_type=1):
    random.seed(seed)
    np.random.seed(seed)
    # dataset_name = ["HASC", ]
    dataset_name = ["Shoaib", ]

    tune_portion = [0.6, 0.8, 1.0]
    for i in range(seg_type):
        for portion in tune_portion:
            for dataset in dataset_name:
                write_balance_tune_set(ori_dir=f'datasets/{dataset}/', train_dir=f'datasets/{dataset}_users_positions_shot{i}/', target_dir=f'datasets/{dataset}_users_positions_tune_portion_{int(portion*100)}_shot{i}/', dataset=dataset, tune_domain_portion=portion, cross='multiple')
    return


if __name__ == '__main__':
    new_segmentation_for_multiple_domain_shift(seg_types=5)