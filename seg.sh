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


##########
# for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
# do
#     name="CDL_wo_negate"
#     python main.py -g 2 -label_type 1 -p2 0 -version "${version}0" -name ${dataset} --store "${name}0" &
#     python main.py -g 2 -label_type 1 -p2 0 -version "${version}1" -name ${dataset} --store "${name}1" &
#     python main.py -g 2 -label_type 1 -p2 0 -version "${version}2" -name ${dataset} --store "${name}2" &
#     python main.py -g 3 -label_type 1 -p2 0 -version "${version}3" -name ${dataset} --store "${name}3" &
#     python main.py -g 3 -label_type 1 -p2 0 -version "${version}4" -name ${dataset} --store "${name}4"

#     wait
# done
shot=10
version="shot"
for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    name="CDL_slr0.7_v"
    ### with all
    store="ewc_results/CDL_slr0.7_all_ewc5_v0"
    python main_trans_ewc.py -shot ${shot} -g 2 -ewc_lambda 5 -aug True -ewc True -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
    python main_trans_ewc.py -shot ${shot} -g 2 -ewc_lambda 5 -aug True -ewc True -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
    python main_trans_ewc.py -shot ${shot} -g 2 -ewc_lambda 5 -aug True -ewc True -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
    python main_trans_ewc.py -shot ${shot} -g 3 -ewc_lambda 5 -aug True -ewc True -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
    python main_trans_ewc.py -shot ${shot} -g 3 -ewc_lambda 5 -aug True -ewc True -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
done
wait
