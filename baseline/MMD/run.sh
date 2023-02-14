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

for shot in 100
do
    version="shot"
    for v in 0 1 2 3 4
    do
        name="CM_cu${v}"
        name2="FM_cu${v}"
        python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HASC --store "${name}" &
        python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name HHAR --store "${name}" &
        python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name MotionSense --store "${name}" &
        python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" -m 'CM' -name Shoaib --store "${name}" &
        python main.py -g 0 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HASC --store "${name2}" &
        python main.py -g 0 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name HHAR --store "${name2}" &
        python main.py -g 0 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name MotionSense --store "${name2}" &
        python main.py -g 0 -version "${version}${v}" -shot ${shot} -cross "users" -m 'FM' -name Shoaib --store "${name2}" 

        wait

    done
done