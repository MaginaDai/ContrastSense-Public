#!/bin/bash
# b=8
# python main.py --store "CV_final_dim_${b}" -name HHAR -g 2 -final_dim ${b}



# version="shot"

# for v in 0 1 2 3 4
# do
#     name="CL_v${v}"
    

#     python main.py -g 0 -label_type 0 -version "${version}${v}" -name HASC --store "${name}" &
#     python main.py -g 0 -label_type 0 -version "${version}${v}" -name HHAR --store "${name}" &
#     python main.py -g 1 -label_type 0 -version "${version}${v}" -name MotionSense --store "${name}" &
#     python main.py -g 1 -label_type 0 -version "${version}${v}" -name Shoaib --store "${name}"

#     wait
#     store="CL_no_ft_design_v${v}"

#     python main_trans_ewc.py -g 0 -ft True -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 0 -ft True -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 1 -ft True -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 1 -ft True -version "${version}${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait

# done


# for v in 0 1 2 3 4
# do
#     name="plain_dim256_v${v}"
#     store="CL_no_aug_v${v}"

#     python main_trans_ewc.py -g 0 -ft True -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 0 -ft True -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 0 -ft True -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 0 -ft True -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait
# done

# version="shot"
# for v in 0 1 2 3 4
# do
#     name="CDL_v${v}"
#     store="CDL_ewc_aug_no_negate_v${v}"

#     python main_trans_ewc.py -g 2 -aug True -ewc True -ft True -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 2 -aug True -ewc True -ft True -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 3 -aug True -ewc True -ft True -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 3 -aug True -ewc True -ft True -version "${version}${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait
# done

# version="cd"
# name="HASC_cd"
# store="HASC_cd"

# python main.py -g 0 -label_type 1 -version "${version}0" -name Shoaib --store "${name}0" -cross "positions" &
# python main.py -g 0 -label_type 1 -version "${version}1" -name Shoaib --store "${name}1" -cross "positions" &
# python main.py -g 1 -label_type 1 -version "${version}2" -name Shoaib --store "${name}2" -cross "positions" &
# python main.py -g 1 -label_type 1 -version "${version}3" -name Shoaib --store "${name}3" -cross "positions" &
# python main.py -g 1 -label_type 1 -version "${version}4" -name Shoaib --store "${name}4" -cross "positions" 

# wait

# python main_trans_ewc.py -g 0 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0/Shoaib" --store "${store}0" -cross "positions" &
# python main_trans_ewc.py -g 0 -aug True -ewc True -ft True -version "${version}1" -shot 10 --pretrained "${name}1/Shoaib" --store "${store}1" -cross "positions" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}2" -shot 10 --pretrained "${name}2/Shoaib" --store "${store}2" -cross "positions" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}3" -shot 10 --pretrained "${name}3/Shoaib" --store "${store}3" -cross "positions" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}4" -shot 10 --pretrained "${name}4/Shoaib" --store "${store}4" -cross "positions" 

# wait


# version="cd"
# name="HASC_cd"
# store="HASC_cd"

# python main.py -g 1 -label_type 1 -version "${version}0" -name HASC --store "${name}0" -cross "devices" &
# python main.py -g 1 -label_type 1 -version "${version}1" -name HASC --store "${name}1" -cross "devices" &
# python main.py -g 1 -label_type 1 -version "${version}2" -name HASC --store "${name}2" -cross "devices" &
# python main.py -g 1 -label_type 1 -version "${version}3" -name HASC --store "${name}3" -cross "devices" &
# python main.py -g 1 -label_type 1 -version "${version}4" -name HASC --store "${name}4" -cross "devices" 

# wait

# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}0" -shot 10 --pretrained "${name}0/HASC" -name HASC --store "${store}0" -cross "devices" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}1" -shot 10 --pretrained "${name}1/HASC" -name HASC --store "${store}1" -cross "devices" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}2" -shot 10 --pretrained "${name}2/HASC" -name HASC --store "${store}2" -cross "devices" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}3" -shot 10 --pretrained "${name}3/HASC" -name HASC --store "${store}3" -cross "devices" &
# python main_trans_ewc.py -g 1 -aug True -ewc True -ft True -version "${version}4" -shot 10 --pretrained "${name}4/HASC" -name HASC --store "${store}4" -cross "devices" 

# wait



# version="shot"
# shot=10
# b=256
# for lam in 0.9
# do
#     for v in 0 1 2 3 4
#     do
#         name="CDL_slr${lam}_v${v}"
#         python main.py -g 0 -b ${b} -slr ${lam} -label_type 1 -version "${version}${v}" -name HASC --store "${name}" &
#         python main.py -g 0 -b ${b} -slr ${lam} -label_type 1 -version "${version}${v}" -name HHAR --store "${name}" &
#         python main.py -g 1 -b ${b} -slr ${lam} -label_type 1 -version "${version}${v}" -name MotionSense --store "${name}" &
#         python main.py -g 1 -b ${b} -slr ${lam} -label_type 1 -version "${version}${v}" -name Shoaib --store "${name}"

#         wait

#         store="CDL_slr${lam}_v${v}"
        
#         python main_trans_ewc.py -g 0 -cl_slr ${lam} -aug True -ewc True -ft True -version "${version}${v}" -shot ${shot} -name HASC --pretrained "${name}/HASC" --store ${store} &
#         python main_trans_ewc.py -g 0 -cl_slr ${lam} -aug True -ewc True -ft True -version "${version}${v}" -shot ${shot} -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#         python main_trans_ewc.py -g 1 -cl_slr ${lam} -aug True -ewc True -ft True -version "${version}${v}" -shot ${shot} -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#         python main_trans_ewc.py -g 1 -cl_slr ${lam} -aug True -ewc True -ft True -version "${version}${v}" -shot ${shot} -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#         wait
#     done
# done


# version="shot"
# shot=10
# for k in 256 512
# do
#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         name="CDL_moco_K${k}_v"
#         python main.py -g 0 -moco_K ${k} -label_type 1 -version "${version}0" -name ${dataset} --store "${name}0" &
#         python main.py -g 0 -moco_K ${k} -label_type 1 -version "${version}1" -name ${dataset} --store "${name}1" &
#         python main.py -g 0 -moco_K ${k} -label_type 1 -version "${version}2" -name ${dataset} --store "${name}2" &
#         python main.py -g 1 -moco_K ${k} -label_type 1 -version "${version}3" -name ${dataset} --store "${name}3" &
#         python main.py -g 1 -moco_K ${k} -label_type 1 -version "${version}4" -name ${dataset} --store "${name}4"
        
#         wait
#     done
# done

# for k in 256 512
# do
#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         name="CDL_moco_K${k}_v"
#         ### with all
#         store="CDL_moco_K${k}_v"
#         python main_trans_ewc.py -g 0 -aug True -ewc True -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
#         python main_trans_ewc.py -g 0 -aug True -ewc True -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
#         python main_trans_ewc.py -g 0 -aug True -ewc True -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
#         python main_trans_ewc.py -g 1 -aug True -ewc True -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
#         python main_trans_ewc.py -g 1 -aug True -ewc True -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
#     done
# done


# shot=50
# for portion in 60 80 100
# do  
#     version="tune_portion_${portion}_shot"
#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         name="CDL_slr0.7_v"
#         ### with all
#         store="CDL_tune_portion_${portion}_v"
#         python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
#         python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
#         python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
#         python main_trans_ewc.py -shot ${shot} -g 1 -aug True -ewc True -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
#         python main_trans_ewc.py -shot ${shot} -g 1 -aug True -ewc True -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
#     done
# done

# shot=10
# version="cd"

# for dataset in "HASC"
# do
#     name="CDL_cd"
#     python main.py -g 0 -label_type 1 -version "${version}0" -name ${dataset} --store "${name}0" -cross "devices" &
#     python main.py -g 0 -label_type 1 -version "${version}1" -name ${dataset} --store "${name}1" -cross "devices" &
#     python main.py -g 0 -label_type 1 -version "${version}2" -name ${dataset} --store "${name}2" -cross "devices" &
#     python main.py -g 1 -label_type 1 -version "${version}3" -name ${dataset} --store "${name}3" -cross "devices" &
#     python main.py -g 1 -label_type 1 -version "${version}4" -name ${dataset} --store "${name}4" -cross "devices" 
    
#     wait
# done


# for dataset in "HASC"
# do
#     name="CDL_cd"
#     ### with all
#     store="CDL_cd"
#     python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
#     python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
#     python main_trans_ewc.py -shot ${shot} -g 0 -aug True -ewc True -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
#     python main_trans_ewc.py -shot ${shot} -g 1 -aug True -ewc True -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
#     python main_trans_ewc.py -shot ${shot} -g 1 -aug True -ewc True -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
# done


# for dataset in "HASC"
# do
#     name="CDL_cd"
#     python main.py -g 0 -label_type 1 -version "${version}0" -name ${dataset} --store "${name}0" -cross "devices" &
#     python main.py -g 0 -label_type 1 -version "${version}1" -name ${dataset} --store "${name}1" -cross "devices" &
#     python main.py -g 0 -label_type 1 -version "${version}2" -name ${dataset} --store "${name}2" -cross "devices" &
#     python main.py -g 1 -label_type 1 -version "${version}3" -name ${dataset} --store "${name}3" -cross "devices" &
#     python main.py -g 1 -label_type 1 -version "${version}4" -name ${dataset} --store "${name}4" -cross "devices" 
    
#     wait
# done

shot=10
version="shot"
##########
for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    name="CL_wo_queue"
    python main.py -g 0 -label_type 0 -moco_K 256 -version "${version}0" -name ${dataset} --store "${name}0" &
    python main.py -g 0 -label_type 0 -moco_K 256 -version "${version}1" -name ${dataset} --store "${name}1" &
    python main.py -g 0 -label_type 0 -moco_K 256 -version "${version}2" -name ${dataset} --store "${name}2" &
    python main.py -g 1 -label_type 0 -moco_K 256 -version "${version}3" -name ${dataset} --store "${name}3" &
    python main.py -g 1 -label_type 0 -moco_K 256 -version "${version}4" -name ${dataset} --store "${name}4"

    wait
done

for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    name="CL_wo_queue"
    ### with all
    store="CL_wo_queue"
    python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" &
    python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" &
    python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" &
    python main_trans_ewc.py -shot ${shot} -g 1 -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" & 
    python main_trans_ewc.py -shot ${shot} -g 1 -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" 
done

wait
