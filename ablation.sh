#!/bin/bash


# name=Shot
# for v in 0 1 2 3 4
# do
    # python main.py -g 0 -label_type 1 -version "shot${v}" -name HASC --store "${name}_${v}_w_HASC" &
    # python main.py -g 0 -label_type 1 -version "shot${v}" -name HHAR --store "${name}_${v}_w_HHAR" &
    # python main.py -g 1 -label_type 1 -version "shot${v}" -name MotionSense --store "${name}_${v}_w_MotionSense" &
    # python main.py -g 1 -label_type 1 -version "shot${v}" -name Shoaib --store "${name}_${v}_w_Shoaib"
    
    # wait
    
    # python main.py -g 0 -label_type 0 -version "shot${v}" -name HASC --store "${name}_${v}_wo_HASC" &
    # python main.py -g 0 -label_type 0 -version "shot${v}" -name HHAR --store "${name}_${v}_wo_HHAR" &
    # python main.py -g 1 -label_type 0 -version "shot${v}" -name MotionSense --store "${name}_${v}_wo_MotionSense" &
    # python main.py -g 1 -label_type 0 -version "shot${v}" -name Shoaib --store "${name}_${v}_wo_Shoaib"

    # wait

#     python main_trans_ewc.py -g 0 -ft True -ewc True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_ewc_pretrain/HASC" --store "f1_change_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -ewc True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_ewc_pretrain/HHAR" --store "f1_change_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -ewc True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_ewc_pretrain/Shoaib" --store "f1_change_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -ewc True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_ewc_pretrain/MotionSense" --store "f1_change_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_ewc_pretrain/HASC" --store "f1_change_wo_ewc_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_ewc_pretrain/HHAR" --store "f1_change_wo_ewc_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_ewc_pretrain/Shoaib" --store "f1_change_wo_ewc_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_ewc_pretrain/MotionSense" --store "f1_change_wo_ewc_${v}"

#     wait

#     python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_wo_HASC" --store "f1_change_wo_CDL_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_wo_HHAR" --store "f1_change_wo_CDL_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_wo_Shoaib" --store "f1_change_wo_CDL_${v}" &
#     python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_wo_MotionSense" --store "f1_change_wo_CDL_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "no" --store "f1_change_no_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "no" --store "f1_change_no_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "no" --store "f1_change_no_${v}" &
#     python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "no" --store "f1_change_no_${v}" 

#     wait
# done


version="shot"
shot=10
slr=0.7
for tem in 0.08 0.09 0.11 0.12
do
    store="hard_v10_cdl_hard_slr0.7_tem${tem}_"
    for dataset in "HASC"
    do
        python main.py -g 2 -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels ${tem} -slr ${slr} -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        python main.py -g 2 -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels ${tem} -slr ${slr} -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        python main.py -g 2 -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels ${tem} -slr ${slr} -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        python main.py -g 3 -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels ${tem} -slr ${slr} -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        python main.py -g 3 -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels ${tem} -slr ${slr} -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        wait
    done

    for dataset in "HASC"
    do
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}0" -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}0" &
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}1" -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}1" &
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}2" -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}2" &
        python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}3" &
        python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}4"
        
        wait
    done

done


