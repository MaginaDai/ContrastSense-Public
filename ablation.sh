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
e=1000
lr=5e-5
for lr in 5e-5
do
    store="emg_cdl_e${e}_lr${lr}_v"
    for dataset in "Myo" "NinaPro"
    do
        # python main_trans_ewc.py -g 2 -ft True -lr 0.0005 -version "shot0" -shot 10 -name ${dataset} --pretrained "no" --store "${store}0" &
        # python main_trans_ewc.py -g 2 -ft True -lr 0.0005 -version "shot1" -shot 10 -name ${dataset} --pretrained "no" --store "${store}1" &
        # python main_trans_ewc.py -g 2 -ft True -lr 0.0005 -version "shot2" -shot 10 -name ${dataset} --pretrained "no" --store "${store}2" &
        # python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "shot3" -shot 10 -name ${dataset} --pretrained "no" --store "${store}3" &
        # python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "shot4" -shot 10 -name ${dataset} --pretrained "no" --store "${store}4"

        # wait

        python main.py -g 0 -e ${e} -lr ${lr} -label_type 1 -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        python main.py -g 0 -e ${e} -lr ${lr} -label_type 1 -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        python main.py -g 1 -e ${e} -lr ${lr} -label_type 1 -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        python main.py -g 1 -e ${e} -lr ${lr} -label_type 1 -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        python main.py -g 1 -e ${e} -lr ${lr} -label_type 1 -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        wait
    done

    for dataset in "Myo" "NinaPro"
    do
        python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "${version}0" -shot 10 -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}0" &
        python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "${version}1" -shot 10 -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}1" &
        python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "${version}2" -shot 10 -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}2" &
        python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "${version}3" -shot 10 -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}3" &
        python main_trans_ewc.py -g 3 -ft True -lr 0.0005 -version "${version}4" -shot 10 -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}4"

        wait
    done
done

