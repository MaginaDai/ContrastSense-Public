#!/bin/bash
# b=8
# python main.py --store "CV_final_dim_${b}" -name HHAR -g 2 -final_dim ${b}

# name="40users"
# for dataset in "HASC" "HHAR" "ICHAR" "MotionSense" "Shoaib" "UCI"
# do
#     python main.py --store "${name}_${dataset}" -name ${dataset} -version '50_200' -g 2 -b 256 -e 500  -label_type 1
#     python main_transfer.py --pretrained "${name}_${dataset}" -g 2 --seed 0 -name ${dataset} -version '50_200' -percent 1  -ft True -lr 0.0005
# done

# name=DAL_lambda
# for b in 0 0.1 0.3 0.5 0.7 0.9 1.0
# do
#     for dataset in 'HASC' 'HHAR' 'MotionSense' ''
#     python main.py --store "${name}_${a}_UCI" -name UCI -version '50_200' -g 2 -b 256 -e 1000  -label_type 1 -slr 0.75 -moco_K 1024
#     python main_transfer.py --pretrained "${name}_${b}_e1000_UCI" -name UCI -version '50_200' -g 2 -ft True -lr 0.0005
# done

# name="DAL_t"

# for dataset in 'HASC' 'HHAR' 'MotionSense' 'Shoaib'
# do 
#     python main.py -g 0 -tem_labels 0.06 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.06_${dataset}" &
#     python main.py -g 0 -tem_labels 0.07 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.07_${dataset}" &
#     python main.py -g 1 -tem_labels 0.08 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.08_${dataset}" &
#     python main.py -g 1 -tem_labels 0.09 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.09_${dataset}" &
#     python main.py -g 1 -tem_labels 0.1 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.1_${dataset}" &
#     python main.py -g 2 -tem_labels 0.2 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.2_${dataset}" &
#     python main.py -g 2 -tem_labels 0.3 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.3_${dataset}" &
#     python main.py -g 3 -tem_labels 0.4 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.4_${dataset}" &
#     python main.py -g 3 -tem_labels 0.5 -label_type 1 -slr 0.5 -DAL True -name ${dataset} --store "${name}_0.5_${dataset}" 
#     wait
# done


# name="Origin_w"
# python main.py -g 1 -tem_labels 0.1 -label_type 1 -slr 0.5 -lr 0.0001 -DAL True -name 'HASC' --store "${name}_HASC" &
# python main.py -g 1 -tem_labels 0.1 -label_type 1 -slr 0.5 -lr 0.0001 -DAL True -name 'HHAR' --store "${name}_HHAR" &
# python main.py -g 2 -tem_labels 0.1 -label_type 1 -slr 0.5 -lr 0.0001 -DAL True -name 'MotionSense' --store "${name}_MotionSense" &
# python main.py -g 2 -tem_labels 0.1 -label_type 1 -slr 0.5 -lr 0.0001 -DAL True -name 'Shoaib' --store "${name}_Shoaib"

# wait

# for t in 0.1
# do
#     python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_HASC" &
#     python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" &
#     python main_transfer.py -g 2 -lr 0.0001 -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" &
#     python main_transfer.py -g 2 -lr 0.0001 -version shot -shot 10 -name MotionSense  --pretrained "${name}_MotionSense"  #######
#     wait
# done

# name="Origin_w"

# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_HHAR" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_Shoaib" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HASC --pretrained "${name}_MotionSense"

# wait

# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HHAR --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HHAR --pretrained "${name}_Shoaib" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name HHAR --pretrained "${name}_MotionSense"

# wait

# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name Shoaib --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name Shoaib --pretrained "${name}_HHAR" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name Shoaib --pretrained "${name}_MotionSense"

# wait

# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name MotionSense --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name MotionSense --pretrained "${name}_HHAR" &
# python main_transfer.py -g 1 -lr 0.0001 -version shot -shot 10 -name MotionSense --pretrained "${name}_Shoaib" &



# name="DAL"
# python main.py -g 1 -name HHAR --store "${name}_HHAR" -DAL True
# python main_transfer.py -g 1 -lr 0.0005 -name HHAR  --pretrained "${name}_HHAR"


######### transfer learning ##########


# name=Origin  # contrastive learning without loss 
# for lr in 0.0001
# do
#     python main.py -g 0 -label_type 0 -lr ${lr} -name HASC --store "${name}_wo_HASC" &
#     python main.py -g 0 -label_type 0 -lr ${lr} -name HHAR --store "${name}_wo_HHAR" &
#     python main.py -g 1 -label_type 0 -lr ${lr} -name MotionSense --store "${name}_wo_MotionSense" &
#     python main.py -g 1 -label_type 0 -lr ${lr} -name Shoaib --store "${name}_wo_Shoaib"

#     wait

#     python main.py -g 0 -label_type 1 -lr ${lr} -name HASC --store "${name}_w_HASC" &
#     python main.py -g 0 -label_type 1 -lr ${lr} -name HHAR --store "${name}_w_HHAR" &
#     python main.py -g 1 -label_type 1 -lr ${lr} -name MotionSense --store "${name}_w_MotionSense" &
#     python main.py -g 1 -label_type 1 -lr ${lr} -name Shoaib --store "${name}_w_Shoaib" &

#     wait

#     python main_transfer.py -g 0 -ft True -lr ${lr} -version shot -shot 10 -name HASC --pretrained "${name}_wo_HASC" &
#     python main_transfer.py -g 0 -ft True -lr ${lr} -version shot -shot 10 -name HHAR --pretrained "${name}_wo_HHAR" &
#     python main_transfer.py -g 0 -ft True -lr ${lr} -version shot -shot 10 -name Shoaib --pretrained "${name}_wo_Shoaib" &
#     python main_transfer.py -g 0 -ft True -lr ${lr} -version shot -shot 10 -name MotionSense  --pretrained "${name}_wo_MotionSense" & #######
#     python main_transfer.py -g 1 -ft True -lr ${lr} -version shot -shot 10 -name HASC --pretrained "${name}_w_HASC" &
#     python main_transfer.py -g 1 -ft True -lr ${lr} -version shot -shot 10 -name HHAR --pretrained "${name}_w_HHAR" &
#     python main_transfer.py -g 1 -ft True -lr ${lr} -version shot -shot 10 -name Shoaib --pretrained "${name}_w_Shoaib" &
#     python main_transfer.py -g 1 -ft True -lr ${lr} -version shot -shot 10 -name MotionSense  --pretrained "${name}_w_MotionSense" & #######
#     python main_transfer.py -g 2 -ft True -lr ${lr} -version shot -shot 10 -name HASC --pretrained "no" &
#     python main_transfer.py -g 2 -ft True -lr ${lr} -version shot -shot 10 -name HHAR --pretrained "no" &
#     python main_transfer.py -g 2 -ft True -lr ${lr} -version shot -shot 10 -name Shoaib --pretrained "no" &
#     python main_transfer.py -g 2 -ft True -lr ${lr} -version shot -shot 10 -name MotionSense  --pretrained "no"

#     wait

# done

# name="Origin_w"
# for slr in 0.9
# do
#     python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name HASC --pretrained "${name}_HASC"  --store "${name}_transfer_DAL_uni_lr1e-4_slr${slr}_bt32" &
#     python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name HHAR --pretrained "${name}_HHAR" --store "${name}_transfer_DAL_uni_lr1e-4_slr${slr}_bt32" &
#     python main_transfer.py -DAL True -g 1 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_transfer_DAL_uni_lr1e-4_slr${slr}_bt32" &
#     python main_transfer.py -DAL True -g 1 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_transfer_DAL_uni_lr1e-4_slr${slr}_bt32"
#     wait
# done

# name="Supervised_portion"
# for s in 1 5 10 15 20 50
# do
#     # python main_transfer.py -g 0 -ft True -lr 0.0001 -version shot -shot ${s} -name HASC --pretrained "${name}_HASC"  --store "${name}_transfer_DAL_uni_slr${slr}_bt32" &
#     python main_transfer.py -g 0 -ft True -lr 0.0001 -version shot -shot ${s} -name HHAR --pretrained "${name}_HHAR" --store "${name}"
#     # python main_transfer.py -g 1 -ft True -lr 0.0001 -version shot -shot ${s} -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_transfer_DAL_uni_slr${slr}_bt32" &
#     # python main_transfer.py -g 1 -ft True -lr 0.0001 -version shot -shot ${s} -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_transfer_DAL_uni_slr${slr}_bt32"
#     # wait
# done


# python main_transfer.py -g 0 -ft True -lr 0.0001 -version shot -shot 1 -name HHAR --pretrained "${name}_HHAR" --store "${name}" &
# python main_transfer.py -g 0 -ft True -lr 0.0001 -version shot -shot 5 -name HHAR --pretrained "${name}_HHAR" --store "${name}" &
# python main_transfer.py -g 0 -ft True -lr 0.0001 -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "${name}" &
# python main_transfer.py -g 1 -ft True -lr 0.0001 -version shot -shot 15 -name HHAR --pretrained "${name}_HHAR" --store "${name}" &
# python main_transfer.py -g 1 -ft True -lr 0.0001 -version shot -shot 20 -name HHAR --pretrained "${name}_HHAR" --store "${name}" &
# python main_transfer.py -g 1 -ft True -lr 0.0001 -version shot -shot 50 -name HHAR --pretrained "${name}_HHAR" --store "${name}"

# name="Origin_w"
# name=ewc_v4

# for fm in 0.01
# do
#     for lam in 1000 2500 7500
#     do
#         python main_trans_ewc.py -ewc_lambda ${lam} -fishermax ${fm} -g 0 -ft True -lr 0.0001 -version "shot" -shot 10 -name HASC --pretrained "${name}_HASC" --store "ewc_v4_fm${fm}_lam${lam}" &
#         python main_trans_ewc.py -ewc_lambda ${lam} -fishermax ${fm} -g 0 -ft True -lr 0.0001 -version "shot" -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "ewc_v4_fm${fm}_lam${lam}" &
#         python main_trans_ewc.py -ewc_lambda ${lam} -fishermax ${fm} -g 1 -ft True -lr 0.0001 -version "shot" -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "ewc_v4_fm${fm}_lam${lam}" &
#         python main_trans_ewc.py -ewc_lambda ${lam} -fishermax ${fm} -g 1 -ft True -lr 0.0001 -version "shot" -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "ewc_v4_fm${fm}_lam${lam}"

#         wait
#     done
# done

# for lr in 0.00005 0.0001 0.0005
# do
#     for e in 200 300 400 500
#     do
#         name='Origin_w'
        
#         python main_transfer.py -g 0 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name HASC --pretrained "${name}_HASC" --store "lr_w_lr${lr}_e${e}" &
#         python main_transfer.py -g 0 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "lr_w_lr${lr}_e${e}" &
#         python main_transfer.py -g 1 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "lr_w_lr${lr}_e${e}" &
#         python main_transfer.py -g 1 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "lr_w_lr${lr}_e${e}"

#         wait

#         name='Origin_wo'

#         python main_transfer.py -g 0 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name HASC --pretrained "${name}_HASC" --store "lr_wo_lr${lr}_e${e}" &
#         python main_transfer.py -g 0 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "lr_wo_lr${lr}_e${e}" &
#         python main_transfer.py -g 1 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "lr_wo_lr${lr}_e${e}" &
#         python main_transfer.py -g 1 -ft True -lr ${lr} -e ${e} -version shot -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "lr_wo_lr${lr}_e${e}"

#         wait
#     done
# done

# name="Origin_w"
# store='trans_SCL'

# for bcl in 128 
# do
#     for slr in 0.7 0.8 0.9 1.0
#     do
#         python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 2 -ft True -lr 0.0005 -version "shot" -shot 10 -name HASC --pretrained "${name}_HASC" --store "${store}_${slr}_bcl_${bcl}" &
#         python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 2 -ft True -lr 0.0005 -version "shot" -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "${store}_${slr}_bcl_${bcl}" &
#         python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 3 -ft True -lr 0.0005 -version "shot" -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "${store}_${slr}_bcl_${bcl}" &
#         python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 3 -ft True -lr 0.0005 -version "shot" -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "${store}_${slr}_bcl_${bcl}"
        
#         wait
#     done
# done


# python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HASC --pretrained "${name}_HASC" --store "ewc_v4" &
# python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "ewc_v4" &
# python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "ewc_v4" &
# python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "ewc_v4"



# name="MoCo_K"
# version="shot"
# v=1
# for k in 1024
# do 
#     name="plain_v${v}"

#     python main.py -g 0 -label_type 0 -version "shot${v}" -name HASC --store "${name}" &
#     python main.py -g 0 -label_type 0 -version "shot${v}" -name HHAR --store "${name}" &
#     python main.py -g 1 -label_type 0 -version "shot${v}" -name MotionSense --store "${name}" &
#     python main.py -g 1 -label_type 0 -version "shot${v}" -name Shoaib --store "${name}"

#     # wait
#     store="MoCo_K${k}_v${v}"

#     python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait
# done


# name="CS_wo"
# version="cross_dataset0"

# python main.py -g 2 -label_type 0 -version "${version}" -name HASC --store "${name}" &
# python main.py -g 2 -label_type 0 -version "${version}" -name HHAR --store "${name}" &
# python main.py -g 3 -label_type 0 -version "${version}" -name MotionSense --store "${name}" &
# python main.py -g 3 -label_type 0 -version "${version}" -name Shoaib --store "${name}"

# wait

# python main_trans_cross_dataset.py -g 2 -ft True -version "${version}" -shot 10 -name HASC --pretrained "${name}/HASC" --store "${name}_wo_CDL" &
# python main_trans_cross_dataset.py -g 2 -ft True -version "${version}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store "${name}_wo_CDL" &
# python main_trans_cross_dataset.py -g 3 -ft True -version "${version}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store "${name}_wo_CDL" &
# python main_trans_cross_dataset.py -g 3 -ft True -version "${version}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store "${name}_wo_CDL_test"

# wait

# name="CS"

# python main_trans_cross_dataset.py -g 2 -ft True -version "${version}" -shot 10 -name HASC --pretrained "${name}/HASC" --store "${name}_wo_ewc" &
# python main_trans_cross_dataset.py -g 2 -ft True -version "${version}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store "${name}_wo_ewc" &
# python main_trans_cross_dataset.py -g 3 -ft True -version "${version}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store "${name}_wo_ewc" &
# python main_trans_cross_dataset.py -g 3 -ft True -version "${version}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store "${name}_wo_ewc"

# wait


version="shot"
v=1

name="plain_v${v}"

python main.py -g 0 -label_type 0 -version "shot${v}" -name HASC --store "${name}" &
python main.py -g 0 -label_type 0 -version "shot${v}" -name HHAR --store "${name}" &
python main.py -g 1 -label_type 0 -version "shot${v}" -name MotionSense --store "${name}" &
python main.py -g 1 -label_type 0 -version "shot${v}" -name Shoaib --store "${name}"

wait
store="plain_v${v}"

python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

name="no"
store="no_v${v}"

python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
python main_trans_ewc.py -g 0 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
python main_trans_ewc.py -g 1 -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 
