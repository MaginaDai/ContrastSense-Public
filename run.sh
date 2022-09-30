#!/bin/bash
# b=8
# python main.py --store "CV_final_dim_${b}" -name HHAR -g 2 -final_dim ${b}

# name="40users"
# for dataset in "HASC" "HHAR" "ICHAR" "MotionSense" "Shoaib" "UCI"
# do
#     python main.py --store "${name}_${dataset}" -name ${dataset} -version '50_200' -g 2 -b 256 -e 500  -label_type 1
#     python main_transfer.py --pretrained "${name}_${dataset}" -g 2 --seed 0 -name ${dataset} -version '50_200' -percent 1  -ft True -lr 0.0005
# done

# name=grid
# for b in 512 1024 2048 3072 4096
# do
#     for a in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do
#         python main.py --store "${name}_queue_${b}_lambda_${a}_UCI" -name UCI -version '50_200' -g 2 -b 256 -e 1000  -label_type 1 -slr 0.75 -moco_K 1024
#         python main_transfer.py --pretrained "${name}_${b}_e1000_UCI" -name UCI -version '50_200' -g 2 -ft True -lr 0.0005
#     done
# done


name="cos"


python main.py -g 1 -e 1000  -label_type 1 -name HASC --store "${name}_HASC" &
python main.py -g 2 -e 1000  -label_type 1 -name HHAR --store "${name}_HHAR" &
python main.py -g 3 -e 1000  -label_type 1 -name Shoaib --store "${name}_Shoaib" 

wait

python main.py -g 1 -e 1000  -label_type 1 -name MotionSense --store "${name}_MotionSense" &
python main_transfer.py -g 2 -ft True -lr 0.0005 -name HASC --pretrained "${name}_HASC"  &
python main_transfer.py -g 3 -ft True -lr 0.0005 -name HHAR --pretrained "${name}_HHAR" 

wait

python main_transfer.py -g 1 -ft True -lr 0.0005 -name Shoaib --pretrained "${name}_Shoaib" &
python main_transfer.py -g 2 -ft True -lr 0.0005 -name MotionSense  --pretrained "${name}_MotionSense"