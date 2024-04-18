#!/bin/bash

# for v in 0 1 2 3 4
# do

#     name='CM_s'
#     python main.py -g 2 -version "s${v}" -m 'CM' -name HASC --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'CM' -name HHAR --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'CM' -name MotionSense --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'CM' -name Shoaib --store "${name}_${v}" 

#     wait

#     name='FM_s'
#     python main.py -g 2 -version "s${v}" -m 'FM' -name HASC --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'FM' -name HHAR --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'FM' -name MotionSense --store "${name}_${v}" &
#     python main.py -g 2 -version "s${v}" -m 'FM' -name Shoaib --store "${name}_${v}" 

#     wait

# done


# g=3
# version="cp"

# name="CM_cp"
# python main.py -g ${g} -version "${version}0" -shot 10 -m 'CM' -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -m 'CM' -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -m 'CM' -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -m 'CM' -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -m 'CM' -name Shoaib --store "${name}4"

# wait

# name="FM_cp"
# python main.py -g ${g} -version "${version}0" -shot 10 -m 'FM' -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -m 'FM' -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -m 'FM' -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -m 'FM' -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -m 'FM' -name Shoaib --store "${name}4"

# wait

# g=3
# version="cd"

# name="CM_cd"
# python main.py -g ${g} -version "${version}0" -shot 10 -cross "devices" -m 'CM' -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -cross "devices" -m 'CM' -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -cross "devices" -m 'CM' -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -cross "devices" -m 'CM' -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -cross "devices" -m 'CM' -name HASC --store "${name}4"

# wait

# name="FM_cd"
# python main.py -g ${g} -version "${version}0" -shot 10 -cross "devices" -m 'FM' -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -cross "devices" -m 'FM' -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -cross "devices" -m 'FM' -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -cross "devices" -m 'FM' -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -cross "devices" -m 'FM' -name HASC --store "${name}4"

# wait

# g=3
# version="shot"

# name="CM_cu"
# python main.py -g ${g} -version "${version}0" -shot 10 -cross "users" -m 'CM' -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -cross "users" -m 'CM' -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -cross "users" -m 'CM' -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -cross "users" -m 'CM' -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -cross "users" -m 'CM' -name HASC --store "${name}4"

# wait

# name="FM_cu"
# python main.py -g ${g} -version "${version}0" -shot 10 -cross "users" -m 'FM' -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -cross "users" -m 'FM' -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -cross "users" -m 'FM' -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -cross "users" -m 'FM' -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -cross "users" -m 'FM' -name HASC --store "${name}4"

# wait

# shot=50
# for portion in 60 80 100
# do
#     version="tune_portion_${portion}_shot"
#     for v in 0 1 2 3 4 
#     do
#         name="CM_cu_tune_portion_${portion}_${v}"
#         name2="FM_cu_tune_portion_${portion}_${v}"
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HASC --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HHAR --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name MotionSense --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name Shoaib --store "${name}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HASC --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HHAR --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name MotionSense --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name Shoaib --store "${name2}" 

#         wait

#     done
# done

# shot=10
# for a in 45 65
# do
#     version="alpha${a}_shot"
#     for v in 0 1 2 3 4
#     do
#         name="CM_alpha${a}_${v}"
#         name2="FM_alpha${a}_${v}"
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HASC --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HHAR --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name MotionSense --store "${name}" &
#         python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name Shoaib --store "${name}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HASC --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HHAR --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name MotionSense --store "${name2}" &
#         python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name Shoaib --store "${name2}" 

#         wait

#     done
# done

# version=cp
# name1="cross positions/CM_cp"
# name2="cross positions/FM_cp"
# g1=2
# g2=3
# for shot in 5 50 100
# do
#     python main.py -g ${g1} -version "${version}0" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}0" &
#     python main.py -g ${g1} -version "${version}1" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}1" &
#     python main.py -g ${g1} -version "${version}2" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}2" &
#     python main.py -g ${g1} -version "${version}3" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}3" &
#     python main.py -g ${g1} -version "${version}4" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}4" &
#     python main.py -g ${g2} -version "${version}0" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}0" &
#     python main.py -g ${g2} -version "${version}1" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}1" &
#     python main.py -g ${g2} -version "${version}2" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}2" &
#     python main.py -g ${g2} -version "${version}3" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}3" &
#     python main.py -g ${g2} -version "${version}4" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}4"
#     wait

# done

# for portion in 60 80 100
# do
#     version="cd_tune_portion_${portion}_shot"
#     name1="cross devices/CM_cd_tune_portion_${portion}_shot"
#     name2="cross devices/FM_cd_tune_portion_${portion}_shot"
#     g1=0
#     g2=1
#     for shot in 50
#     do
#         python main.py -g ${g1} -version "${version}0" -shot ${shot} -m 'CM' -cross "devices" -name HASC --store "${name1}0" &
#         python main.py -g ${g1} -version "${version}1" -shot ${shot} -m 'CM' -cross "devices" -name HASC --store "${name1}1" &
#         python main.py -g ${g1} -version "${version}2" -shot ${shot} -m 'CM' -cross "devices" -name HASC --store "${name1}2" &
#         python main.py -g ${g1} -version "${version}3" -shot ${shot} -m 'CM' -cross "devices" -name HASC --store "${name1}3" &
#         python main.py -g ${g1} -version "${version}4" -shot ${shot} -m 'CM' -cross "devices" -name HASC --store "${name1}4" &
#         python main.py -g ${g2} -version "${version}0" -shot ${shot} -m 'FM' -cross "devices" -name HASC --store "${name2}0" &
#         python main.py -g ${g2} -version "${version}1" -shot ${shot} -m 'FM' -cross "devices" -name HASC --store "${name2}1" &
#         python main.py -g ${g2} -version "${version}2" -shot ${shot} -m 'FM' -cross "devices" -name HASC --store "${name2}2" &
#         python main.py -g ${g2} -version "${version}3" -shot ${shot} -m 'FM' -cross "devices" -name HASC --store "${name2}3" &
#         python main.py -g ${g2} -version "${version}4" -shot ${shot} -m 'FM' -cross "devices" -name HASC --store "${name2}4"
#         wait
#     done
# done

# version="cp_tune_portion_100_shot"
# name1="cross positions/CM_cp_tune_portion_100_shot"
# name2="cross positions/FM_cp_tune_portion_100_shot"
# g1=0
# g2=1
# for shot in 5 10 50 100
# do
#     python main.py -g ${g1} -version "${version}0" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}0" &
#     python main.py -g ${g1} -version "${version}1" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}1" &
#     python main.py -g ${g1} -version "${version}2" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}2" &
#     python main.py -g ${g1} -version "${version}3" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}3" &
#     python main.py -g ${g1} -version "${version}4" -shot ${shot} -m 'CM' -cross "positions" -name Shoaib --store "${name1}4" &
#     python main.py -g ${g2} -version "${version}0" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}0" &
#     python main.py -g ${g2} -version "${version}1" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}1" &
#     python main.py -g ${g2} -version "${version}2" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}2" &
#     python main.py -g ${g2} -version "${version}3" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}3" &
#     python main.py -g ${g2} -version "${version}4" -shot ${shot} -m 'FM' -cross "positions" -name Shoaib --store "${name2}4"
#     wait

# done

# version=train25_supervised_cross
# g1=0
# name1=preliminary_25_cross_FM

# python main.py -g ${g1} -version "${version}" -shot 0 -m 'FM' -cross "users" -name HHAR --store "${name1}" &
# python main.py -g ${g1} -version "${version}" -shot 5 -m 'FM' -cross "users" -name HHAR --store "${name1}" &
# python main.py -g ${g1} -version "${version}" -shot 10 -m 'FM' -cross "users" -name HHAR --store "${name1}" &
# python main.py -g ${g1} -version "${version}" -shot 50 -m 'FM' -cross "users" -name HHAR --store "${name1}" &
# python main.py -g ${g1} -version "${version}" -shot 100 -m 'FM' -cross "users" -name HHAR --store "${name1}"


# shot=10

# version="alpha${a}_shot"
# for v in 0 1 2 3 4
# do
#     name="CM_alpha${a}_${v}"
#     name2="FM_alpha${a}_${v}"
#     python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HASC --store "${name}" &
#     python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HHAR --store "${name}" &
#     python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name MotionSense --store "${name}" &
#     python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name Shoaib --store "${name}" &
#     python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HASC --store "${name2}" &
#     python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HHAR --store "${name2}" &
#     python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name MotionSense --store "${name2}" &
#     python main.py -g 3 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name Shoaib --store "${name2}" 

#     wait

# done


# version="users_devices_shot"
# name="CM_users_devices_"
# name2="FM_users_devices_"

# for shot in 50 100
# do
#     python main.py -g 2 -version "${version}0" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}0" &
#     python main.py -g 2 -version "${version}1" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}1" &
#     python main.py -g 2 -version "${version}2" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}2" &
#     python main.py -g 2 -version "${version}3" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}3" &
#     python main.py -g 2 -version "${version}4" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}4" &
#     python main.py -g 3 -version "${version}0" -shot ${shot} -cross "multiple" -m 'FM' -name HASC --store "${name2}0" &
#     python main.py -g 3 -version "${version}1" -shot ${shot} -cross "multiple" -m 'FM' -name HASC --store "${name2}1" &
#     python main.py -g 3 -version "${version}2" -shot ${shot} -cross "multiple" -m 'FM' -name HASC --store "${name2}2" &
#     python main.py -g 3 -version "${version}3" -shot ${shot} -cross "multiple" -m 'FM' -name HASC --store "${name2}3" &
#     python main.py -g 3 -version "${version}4" -shot ${shot} -cross "multiple" -m 'FM' -name HASC --store "${name2}4" 
#     wait

# done


version="leave_shot"
name="CM_leave_shot"
name2="FM_leave_shot"
shot=10
for dataset in "MotionSense"
do
    python main.py -g 0 -version "${version}0" -shot ${shot} -cross "users" -m 'CM' -name ${dataset} --store "${name}0" &
    python main.py -g 0 -version "${version}1" -shot ${shot} -cross "users" -m 'CM' -name ${dataset} --store "${name}1" &
    python main.py -g 0 -version "${version}2" -shot ${shot} -cross "users" -m 'CM' -name ${dataset} --store "${name}2" &
    python main.py -g 0 -version "${version}3" -shot ${shot} -cross "users" -m 'CM' -name ${dataset} --store "${name}3" &
    python main.py -g 0 -version "${version}4" -shot ${shot} -cross "users" -m 'CM' -name ${dataset} --store "${name}4" &
    python main.py -g 0 -version "${version}0" -shot ${shot} -cross "users" -m 'FM' -name ${dataset} --store "${name2}0" &
    python main.py -g 0 -version "${version}1" -shot ${shot} -cross "users" -m 'FM' -name ${dataset} --store "${name2}1" &
    python main.py -g 1 -version "${version}2" -shot ${shot} -cross "users" -m 'FM' -name ${dataset} --store "${name2}2" &
    python main.py -g 1 -version "${version}3" -shot ${shot} -cross "users" -m 'FM' -name ${dataset} --store "${name2}3" &
    python main.py -g 1 -version "${version}4" -shot ${shot} -cross "users" -m 'FM' -name ${dataset} --store "${name2}4" 
    wait

done
