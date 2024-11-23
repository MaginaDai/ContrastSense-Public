import sys
import os
from os.path import dirname
sys.path.append(dirname(dirname(sys.path[0])))

from preprocessing import datasets_shot_record, preprocessing_dataset_cross_person_val


path_save = r'./datasets/HASC_50_200/'
preprocessing_dataset_cross_person_val(dir=path_save, dataset='HASC', dataset_size=10291, test_portion=0.5)
datasets_shot_record(dir=path_save, datasets='HASC')