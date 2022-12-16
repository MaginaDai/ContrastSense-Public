#!/bin/bash


name=Shot
for v in 0 1 2 3 4
do
    python main.py -g 0 -label_type 1 -version "shot${v}" -name HASC --store "${name}_${v}_w_HASC" &
    python main.py -g 0 -label_type 1 -version "shot${v}" -name HHAR --store "${name}_${v}_w_HHAR" &
    python main.py -g 1 -label_type 1 -version "shot${v}" -name MotionSense --store "${name}_${v}_w_MotionSense" &
    python main.py -g 1 -label_type 1 -version "shot${v}" -name Shoaib --store "${name}_${v}_w_Shoaib"
    
    wait
    
    python main.py -g 0 -label_type 0 -version "shot${v}" -name HASC --store "${name}_${v}_wo_HASC" &
    python main.py -g 0 -label_type 0 -version "shot${v}" -name HHAR --store "${name}_${v}_wo_HHAR" &
    python main.py -g 1 -label_type 0 -version "shot${v}" -name MotionSense --store "${name}_${v}_wo_MotionSense" &
    python main.py -g 1 -label_type 0 -version "shot${v}" -name Shoaib --store "${name}_${v}_wo_Shoaib"

    wait

    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_wo_HASC" --store "${name}_${v}_wo_lr5e-4" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_wo_HHAR" --store "${name}_${v}_wo_lr5e-4" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_wo_Shoaib" --store "${name}_${v}_wo_lr5e-4" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_wo_MotionSense" --store "${name}_${v}_wo_lr5e-4" 

    wait

    python main_transfer.py -g 1 -DAL True -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_w_HASC" --store "${name}_${v}_DAL0.9_lr5e-4" &
    python main_transfer.py -g 1 -DAL True -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_w_HHAR" --store "${name}_${v}_DAL0.9_lr5e-4" &
    python main_transfer.py -g 1 -DAL True -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_w_Shoaib" --store "${name}_${v}_DAL0.9_lr5e-4" &
    python main_transfer.py -g 1 -DAL True -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_w_MotionSense" --store "${name}_${v}_DAL0.9_lr5e-4" 

    wait

    python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HASC --pretrained "${name}_${v}_w_HASC" --store "ewc_v4_e400_${v}" &
    python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HHAR --pretrained "${name}_${v}_w_HHAR" --store "ewc_v4_e400_${v}" &
    python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name Shoaib --pretrained "${name}_${v}_w_Shoaib" --store "ewc_v4_e400_${v}" &
    python main_trans_ewc.py -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name MotionSense  --pretrained "${name}_${v}_w_MotionSense" --store "ewc_v4_e400_${v}"

    wait

    python main_transfer.py -g 0 -ft True -lr 0.0001 -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_wo_HASC" &
    python main_transfer.py -g 0 -ft True -lr 0.0001 -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_wo_HHAR" &
    python main_transfer.py -g 1 -ft True -lr 0.0001 -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_wo_Shoaib" &
    python main_transfer.py -g 1 -ft True -lr 0.0001 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_wo_MotionSense"

    wait

    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HASC --pretrained "no" --store "${name}_${v}_no" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name HHAR --pretrained "no" --store "${name}_${v}_no" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name Shoaib --pretrained "no" --store "${name}_${v}_no" &
    python main_transfer.py -g 1 -ft True -lr 0.0005 -version "shot${v}" -shot 10 -name MotionSense  --pretrained "no" --store "${name}_${v}_no" 

    wait
done