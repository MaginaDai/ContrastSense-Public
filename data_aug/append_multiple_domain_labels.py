import numpy as np
from preprocessing import fetch_instance_number_of_dataset


def append_labels(dir, dataset):
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
    for i in range(num):
        sub_dir = dir + str(i) + '.npz'
        multi_domain_label = domain[i]
        for n, ds in enumerate(domain_type):
            if np.all(multi_domain_label == ds):
                multi_domain_label = n
                data = np.load(sub_dir, allow_pickle=True)
                infor = data['add_infor']
                infor=np.insert(infor, 3, multi_domain_label)
                np.savez(sub_dir, acc=data['acc'], gyro=data['gyro'], add_infor=infor)
                break
            
    return

if __name__ == '__main__':
    dataset='HASC'
    append_labels(dir=f'datasets/{dataset}/', dataset=dataset)