#!/bin/bash

# num=15

# cp -r 'datasets/HHAR_50_200/' "datasets/HHAR_50_200_${num}"
# cp -r 'datasets/MotionSense_50_200/' "datasets/MotionSense_50_200_${num}"
# cp -r 'datasets/Shoaib_50_200/' "datasets/Shoaib_50_200_${num}"
# cp -r 'datasets/HASC_50_200/' "datasets/HASC_50_200_${num}"

# python data_aug/shoaib.py

# python data_aug/UCI.py

# python data_aug/MotionSense_Prep.py

# python data_aug/ichar.py

# python data_aug/preprocessing.py
# name="DAL"
# python main.py -g 2 -label_type 1 -DAL True -name HASC --store "${name}_slr0.4_HASC" &
# python main.py -g 2 -label_type 1 -DAL True -name HASC --store "${name}_slr0.8_HASC"

# wait

# python main_transfer.py -g 2 -ft True -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_slr0.4_HASC" &
# python main_transfer.py -g 2 -ft True -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_slr0.8_HASC"
# name="Origin_wo"

# python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -ad-lr 0.0000005 -name HHAR --pretrained "${name}_HHAR" --store "Origin_wo_transfer_DAL_lr0.0000005_sep" &
# python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -ad-lr 0.00000005 -name HHAR --pretrained "${name}_HHAR" --store "Origin_wo_transfer_DAL_lr0.00000005_sep" &
