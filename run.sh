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


# a=0.3
# for e in 2000
# do 
#     name="DAL"
#     namea="CE_2000_cos_trans"
#     nameb="contrast_wo"
#     namec="no"
    # python main.py -g 0 -e ${e}  -label_type 1 -slr ${a} -CE True -name MotionSense --store "${name}_MotionSense" &
    # python main.py -g 0 -e ${e}  -label_type 1 -slr ${a} -CE True -name HASC --store "${name}_HASC" &
    # python main.py -g 1 -e ${e}  -label_type 1 -slr ${a} -CE True -name HHAR --store "${name}_HHAR" &
    # python main.py -g 1 -e ${e}  -label_type 1 -slr ${a} -CE True -name Shoaib --store "${name}_Shoaib" 

    # wait

    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HASC --pretrained "${name}_HASC" --store "${name}_1_HASC" &
    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HHAR --pretrained "${name}_HHAR" --store "${name}_1_HHAR" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_1_Shoaib" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_1_MotionSense" 

    # wait


    # python main.py -g 0 -e ${e}  -label_type 0 -slr ${a} -name MotionSense --store "${name}_MotionSense" &
    # python main.py -g 0 -e ${e}  -label_type 0 -slr ${a} -name HASC --store "${name}_HASC" &
    # python main.py -g 1 -e ${e}  -label_type 0 -slr ${a} -name HHAR --store "${name}_HHAR" &
    # python main.py -g 1 -e ${e}  -label_type 0 -slr ${a} -name Shoaib --store "${name}_Shoaib" 

    # wait

    # wait



    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HASC --pretrained "${nameb}_HASC" --store "${nameb}_cos_trans_HASC" &
    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HHAR --pretrained "${nameb}_HHAR" --store "${nameb}_cos_trans_HHAR" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name Shoaib --pretrained "${nameb}_Shoaib" --store "${nameb}_cos_trans_Shoaib" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name MotionSense  --pretrained "${nameb}_MotionSense" --store "${nameb}_cos_trans_MotionSense"

    # wait
    
    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HASC --pretrained "${namec}_HASC" &
    # python main_transfer.py -g 0 -ft True -lr 0.0005 -name HHAR --pretrained "${namec}_HHAR" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name Shoaib --pretrained "${namec}_Shoaib" &
    # python main_transfer.py -g 1 -ft True -lr 0.0005 -name MotionSense  --pretrained "${namec}_MotionSense"


# done
# name="DAL"
# python main.py -g 1 -name HHAR --store "${name}_HHAR" -DAL True
# python main_transfer.py -g 1 -lr 0.0005 -name HHAR  --pretrained "${name}_HHAR"


######### transfer learning ##########

name=contrast_w  # contrastive learning without loss 
for lr in 0.001 0.005
do
    python main_transfer.py -g 1 -ft True -lr ${lr} -version 50_200_shot -shot 10 -name HASC --pretrained "${name}_HASC" --store "${name}_HASC_shot_lr_${lr}" &
    python main_transfer.py -g 1 -ft True -lr ${lr} -version 50_200_shot -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "${name}_HHAR_shot_lr_${lr}" &
    python main_transfer.py -g 1 -ft True -lr ${lr} -version 50_200_shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_Shoaib_shot_lr_${lr}" &
    python main_transfer.py -g 1 -ft True -lr ${lr} -version 50_200_shot -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_MotionSense_shot_lr_${lr}"
    wait
done

# wait

# name=no  # contrastive learning without loss 
# for shot in 1 5 10 15 20 50
# do
#     python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -shot ${shot} -name HASC --pretrained "${name}_HASC" --store "${name}_HASC_shot" &
#     python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -shot ${shot} -name HHAR --pretrained "${name}_HHAR" --store "${name}_HHAR_shot" &
#     python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -shot ${shot} -name Shoaib --pretrained "${name}_Shoaib" --store "${name}_Shoaib_shot" &
#     python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -shot ${shot} -name MotionSense  --pretrained "${name}_MotionSense" --store "${name}_MotionSense_shot"
#     wait
# done

# wait
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HASC --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HHAR --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name Shoaib --pretrained "${name}_HASC" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name MotionSense  --pretrained "${name}_HASC"
# wait
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HASC --pretrained "${name}_Shoaib" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HHAR --pretrained "${name}_Shoaib" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name Shoaib --pretrained "${name}_Shoaib" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name MotionSense  --pretrained "${name}_Shoaib"
# wait
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HASC --pretrained "${name}_MotionSense" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name HHAR --pretrained "${name}_MotionSense" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name Shoaib --pretrained "${name}_MotionSense" &
# python main_transfer.py -g 1 -ft True -lr 0.0005 -version 50_200_shot -name MotionSense  --pretrained "${name}_MotionSense"
