import os
import numpy as np

source1=r'./datasets/HHAR_test/'
source2=r'./datasets/HHAR/'

file_name_list = [file for file in os.listdir(source1) if 'set' not in file]
num = len(file_name_list)
print(num)
total = 0
for idx in np.arange(num):
    loc1 = os.path.join(source1, f'{idx}.npz')
    sample1 = np.load(loc1, allow_pickle=True)
    acc1, gyro1, add_infor = sample1['acc'], sample1['gyro'], sample1['add_infor']
    
    loc2 = os.path.join(source2, f'{idx}.npz')
    sample2 = np.load(loc2, allow_pickle=True)
    acc2, gyro2, add_infor = sample2['acc'], sample2['gyro'], sample2['add_infor']

    total += np.sum(acc1 - acc2) + np.sum(gyro1 - gyro2)

print(total)