# name='CPC_s'
# g=0
# version=shot

# for v in 0 1 2 3 4
# do
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name HASC &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name HHAR &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name MotionSense &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name Shoaib 

#     wait

#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}${v}/HASC" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}${v}/HHAR" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name MotionSense --pretrained "${name}${v}/MotionSense" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}${v}/Shoaib" --store "${name}${v}" 

#     wait

# done

# name='CPCHAR_cp'
# version="cp"
# g=2

# python main.py -g ${g} -version "${version}0" -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -name Shoaib --store "${name}4" &

# wait

# python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name Shoaib --pretrained "${name}0/Shoaib" --store "${name}0" &
# python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name Shoaib --pretrained "${name}1/Shoaib" --store "${name}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name Shoaib --pretrained "${name}2/Shoaib" --store "${name}2" &
# python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name Shoaib --pretrained "${name}3/Shoaib" --store "${name}3" &
# python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name Shoaib --pretrained "${name}4/Shoaib" --store "${name}4" 

# wait

# name='CPCHAR_cd'
# version="cd"
# g=2

# python main.py -g ${g} -version "${version}0" -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -name HASC --store "${name}4" &

# wait

# python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name HASC --pretrained "${name}0/HASC" --store "${name}0" &
# python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name HASC --pretrained "${name}1/HASC" --store "${name}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name HASC --pretrained "${name}2/HASC" --store "${name}2" &
# python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name HASC --pretrained "${name}3/HASC" --store "${name}3" &
# python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name HASC --pretrained "${name}4/HASC" --store "${name}4" 

# wait

# name='CPCHAR_cu'
# version="shot"
# g=2

# python main.py -g ${g} -version "${version}0" -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -name HASC --store "${name}4" &

# wait

# python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name HASC --pretrained "${name}0/HASC" --store "${name}0" &
# python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name HASC --pretrained "${name}1/HASC" --store "${name}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name HASC --pretrained "${name}2/HASC" --store "${name}2" &
# python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name HASC --pretrained "${name}3/HASC" --store "${name}3" &
# python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name HASC --pretrained "${name}4/HASC" --store "${name}4" 

# wait


# name='CPCHAR_cu'
# g=2
# shot=50
# for portion in 60 80 100
# do  
#     version="tune_portion_${portion}_shot"
#     for v in 0 1 2 3 4
#     do
#         store="CPCHAR_cu_tune_portion_${portion}_${v}"
#         python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name HASC --pretrained "${name}${v}/HASC" --store "${store}" &
#         python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name HHAR --pretrained "${name}${v}/HHAR" --store "${store}" &
#         python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name MotionSense --pretrained "${name}${v}/MotionSense" --store "${store}" &
#         python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name Shoaib --pretrained "${name}${v}/Shoaib" --store "${store}" 

#         wait
#     done
# done

name='CPCHAR_cu'
g=2
for shot in 100
do  
    version="shot"
    for v in 0 1 2 3 4
    do
        store="CPCHAR_cu_tune_portion_${portion}_${v}"
        python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name HASC --pretrained "${name}${v}/HASC" --store "${store}" &
        python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name HHAR --pretrained "${name}${v}/HHAR" --store "${store}" &
        python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name MotionSense --pretrained "${name}${v}/MotionSense" --store "${store}" &
        python transfer.py -g ${g} -ft True -version "${version}${v}" -shot ${shot} -name Shoaib --pretrained "${name}${v}/Shoaib" --store "${store}" 

        wait
    done
done