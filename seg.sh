#!/bin/bash

num=15

cp -r 'datasets/HHAR_50_200/' "datasets/HHAR_50_200_${num}"
cp -r 'datasets/MotionSense_50_200/' "datasets/MotionSense_50_200_${num}"
cp -r 'datasets/Shoaib_50_200/' "datasets/Shoaib_50_200_${num}"
cp -r 'datasets/HASC_50_200/' "datasets/HASC_50_200_${num}"

# python data_aug/shoaib.py

# python data_aug/UCI.py

# python data_aug/MotionSense_Prep.py

# python data_aug/ichar.py

# python data_aug/preprocessing.py

