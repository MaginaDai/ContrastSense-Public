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

# wait

# name=BN_no  # contrastive learning without loss 
# for shot in 10
# do
#     python main_transfer.py -g 1 -ft True -lr 0.001 -version 50_200_shot -shot ${shot} -name HASC --pretrained "${name}_HASC" --store "${name}_HASC" &
#     python main_transfer.py -g 1 -ft True -lr 0.001 -version 50_200_shot -shot ${shot} -name HHAR --pretrained "${name}_HHAR" --store "${name}_HHAR" &
#     python main_transfer.py -g 1 -ft True -lr 0.001 -version 50_200_shot -shot ${shot} -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_Shoaib" &
#     python main_transfer.py -g 1 -ft True -lr 0.001 -version 50_200_shot -shot ${shot} -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_MotionSense"
#     wait
# done

# wait


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

name="Origin_w"
for slr in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name HASC --pretrained "${name}_HASC"  --store "${name}_transfer_DAL_uni_slr${slr}_bt32" &
    python main_transfer.py -DAL True -g 0 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name HHAR --pretrained "${name}_HHAR" --store "${name}_transfer_DAL_uni_slr${slr}_bt32" &
    python main_transfer.py -DAL True -g 1 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_transfer_DAL_uni_slr${slr}_bt32" &
    python main_transfer.py -DAL True -g 1 -ft True -lr 0.0001 -version shot -shot 10 -slr ${slr} -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_transfer_DAL_uni_slr${slr}_bt32"
    wait
done
