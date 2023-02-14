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


# version="shot"

# for v in 3 4
# do
#     name="CDL_v${v}"
    

#     python main.py -g 2 -label_type 1 -version "${version}${v}" -name HASC --store "${name}" &
#     python main.py -g 2 -label_type 1 -version "${version}${v}" -name HHAR --store "${name}" &
#     python main.py -g 3 -label_type 1 -version "${version}${v}" -name MotionSense --store "${name}" &
#     python main.py -g 3 -label_type 1 -version "${version}${v}" -name Shoaib --store "${name}"

#     wait

#     store="CDL_no_ft_design_v${v}"

#     python main_trans_ewc.py -g 2 -ft True -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 2 -ft True -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 3 -ft True -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 3 -ft True -version "${version}${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait

# done


# version="cd"
# name="cd_slr"
# store="cd_slr"

# python main.py -g 0 -label_type 1 -slr 0.6 -version "${version}0" -name HHAR --store "${name}0.6" -cross "devices" &
# python main.py -g 0 -label_type 1 -slr 0.7 -version "${version}0" -name HHAR --store "${name}0.7" -cross "devices" &
# python main.py -g 1 -label_type 1 -slr 0.8 -version "${version}0" -name HHAR --store "${name}0.8" -cross "devices" &
# python main.py -g 1 -label_type 1 -slr 0.9 -version "${version}0" -name HHAR --store "${name}0.9" -cross "devices" &
# python main.py -g 1 -label_type 1 -slr 1.0 -version "${version}0" -name HHAR --store "${name}1.0" -cross "devices" 

# wait

# python main_trans_ewc.py -g 0 -cl_slr 0.6 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0.6/HHAR" -name HHAR --store "${store}0.6" -cross "devices" &
# python main_trans_ewc.py -g 0 -cl_slr 0.7 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0.7/HHAR" -name HHAR --store "${store}0.7" -cross "devices" &
# python main_trans_ewc.py -g 1 -cl_slr 0.8 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0.8/HHAR" -name HHAR --store "${store}0.8" -cross "devices" &
# python main_trans_ewc.py -g 1 -cl_slr 0.9 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0.9/HHAR" -name HHAR --store "${store}0.9" -cross "devices" &
# python main_trans_ewc.py -g 1 -cl_slr 1.0 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}1.0/HHAR" -name HHAR --store "${store}1.0" -cross "devices" 

# wait

# version="shot"
# shot=10
# lam=1

# for v in 3 4
# do
#     name="CDL_v${v}"
#     store="CDL_ewc_pretrain_v${v}"

#     python main_trans_ewc.py -g 1 -ewc_lambda 1 -ewc True -ft True -version "${version}${v}" -shot ${shot} -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 1 -ewc_lambda 1 -ewc True -ft True -version "${version}${v}" -shot ${shot} -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 2 -ewc_lambda 1 -ewc True -ft True -version "${version}${v}" -shot ${shot} -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 2 -ewc_lambda 1 -ewc True -ft True -version "${version}${v}" -shot ${shot} -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait
# done

shot=10
version="shot"
##########
for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    name="CDL_wo_scale"
    python main.py -g 2 -label_type 1 -p6 0 -version "${version}0" -name ${dataset} --store "${name}0" &
    python main.py -g 2 -label_type 1 -p6 0 -version "${version}1" -name ${dataset} --store "${name}1" &
    python main.py -g 2 -label_type 1 -p6 0 -version "${version}2" -name ${dataset} --store "${name}2" &
    python main.py -g 3 -label_type 1 -p6 0 -version "${version}3" -name ${dataset} --store "${name}3" &
    python main.py -g 3 -label_type 1 -p6 0 -version "${version}4" -name ${dataset} --store "${name}4"

    wait
done

for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    name="CDL_wo_scale"
    ### with all
    store="CDL_wo_scale"
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
    python main_trans_ewc.py -shot ${shot} -g 3 -aug True -ewc True -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
    python main_trans_ewc.py -shot ${shot} -g 3 -aug True -ewc True -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
done
wait
