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
# g=1

# python main.py -g ${g} -version "${version}0" -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -name Shoaib --store "${name}4" &

# wait
# for shot in 5 50 100
# do
# python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name Shoaib --pretrained "${name}0/Shoaib" --store "${name}0" &
# python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name Shoaib --pretrained "${name}1/Shoaib" --store "${name}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name Shoaib --pretrained "${name}2/Shoaib" --store "${name}2" &
# python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name Shoaib --pretrained "${name}3/Shoaib" --store "${name}3" &
# python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name Shoaib --pretrained "${name}4/Shoaib" --store "${name}4" 
# wait

# done

# g=1
# for portion in 60 80 100
# do
#     version="cd_tune_portion_${portion}_shot"
#     name="CPCHAR_cd"
#     store="CPCHAR_cd_tune_portion_${portion}_shot"
#     for shot in 50
#     do
#         python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name HASC --pretrained "${name}0/HASC" --store "${store}0" &
#         python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name HASC --pretrained "${name}1/HASC" --store "${store}1" &
#         python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name HASC --pretrained "${name}2/HASC" --store "${store}2" &
#         python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name HASC --pretrained "${name}3/HASC" --store "${store}3" &
#         python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name HASC --pretrained "${name}4/HASC" --store "${store}4" 
#     wait
# done
# done

# g=1
# portion=100
# version="cp_tune_portion_${portion}_shot"
# name="CPCHAR_cp"
# store="CPCHAR_cp_tune_portion_${portion}_shot"

# python transfer.py -g ${g} -ft True -version "${version}1" -shot 50 -name Shoaib --pretrained "${name}1/Shoaib" --store "${store}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot 100 -name Shoaib --pretrained "${name}2/Shoaib" --store "${store}2" &

# for shot in 5 10 50 100
# do
#     python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name Shoaib --pretrained "${name}0/Shoaib" --store "${store}0" &
#     python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name Shoaib --pretrained "${name}1/Shoaib" --store "${store}1" &
#     python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name Shoaib --pretrained "${name}2/Shoaib" --store "${store}2" &
#     python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name Shoaib --pretrained "${name}3/Shoaib" --store "${store}3" &
#     python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name Shoaib --pretrained "${name}4/Shoaib" --store "${store}4" 
#     wait
# done

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

# name='CPCHAR_cu'
# g=2
# for shot in 1
# do  
#     version="shot"
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

# g=2
# version="train25_supervised_label"
# name="CPCHAR_preliminary"
# store="CPCHAR_preliminary"

# python main.py -g ${g} -version "${version}" -name HHAR --store "${name}"
# wait

# python transfer.py -g ${g} -ft True -version "${version}" -shot 0 -name HHAR --pretrained "${name}/HHAR" --store "${store}" &
# python transfer.py -g ${g} -ft True -version "${version}" -shot 5 -name HHAR --pretrained "${name}/HHAR" --store "${store}" &
# python transfer.py -g ${g} -ft True -version "${version}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store "${store}" &
# python transfer.py -g ${g} -ft True -version "${version}" -shot 50 -name HHAR --pretrained "${name}/HHAR" --store "${store}" &
# python transfer.py -g ${g} -ft True -version "${version}" -shot 100 -name HHAR --pretrained "${name}/HHAR" --store "${store}" 


# g=2
# version="train25_supervised_label"
# name="CPCHAR_preliminary"
# store="CPCHAR_preliminary"

# python main.py -g 2 -version "train25_supervised_random" -name HHAR --store "25_random" &
# python main.py -g 2 -version "train45_supervised_random" -name HHAR --store "45_random" &
# python main.py -g 2 -version "train65_supervised_random" -name HHAR --store "65_random" &
# python main.py -g 3 -version "train25_supervised_label" -name HHAR --store "25_label" &
# python main.py -g 3 -version "train45_supervised_label" -name HHAR --store "45_label" &
# python main.py -g 3 -version "train65_supervised_label" -name HHAR --store "65_label" &

# wait

# python transfer.py -g 2 -ft True -version "train25_supervised_random" -shot 50 -name HHAR --pretrained "25_random/HHAR" --store "25_random" &
# python transfer.py -g 2 -ft True -version "train45_supervised_random" -shot 50 -name HHAR --pretrained "45_random/HHAR" --store "45_random" &
# python transfer.py -g 2 -ft True -version "train65_supervised_random" -shot 50 -name HHAR --pretrained "65_random/HHAR" --store "65_random" &
# python transfer.py -g 3 -ft True -version "train25_supervised_label" -shot 50 -name HHAR --pretrained "25_label/HHAR" --store "25_label" &
# python transfer.py -g 3 -ft True -version "train45_supervised_label" -shot 50 -name HHAR --pretrained "45_label/HHAR" --store "45_label" &
# python transfer.py -g 3 -ft True -version "train65_supervised_label" -shot 50 -name HHAR --pretrained "65_label/HHAR" --store "65_label" 

# python transfer.py -g 2 -ft True -version "train25_supervised_random" -shot 50 -name HHAR --pretrained "random/HHAR" --store "25_random_sup" &
# python transfer.py -g 2 -ft True -version "train45_supervised_random" -shot 50 -name HHAR --pretrained "random/HHAR" --store "45_random_sup" &
# python transfer.py -g 2 -ft True -version "train65_supervised_random" -shot 50 -name HHAR --pretrained "random/HHAR" --store "65_random_sup" &
# python transfer.py -g 3 -ft True -version "train25_supervised_label" -shot 50 -name HHAR --pretrained "random/HHAR" --store "25_label_sup" &
# python transfer.py -g 3 -ft True -version "train45_supervised_label" -shot 50 -name HHAR --pretrained "random/HHAR" --store "45_label_sup" &
# python transfer.py -g 3 -ft True -version "train65_supervised_label" -shot 50 -name HHAR --pretrained "random/HHAR" --store "65_label_sup" 


for a in 45 65
do
    name="CPCHAR_alpha${a}_"
    version="alpha${a}_shot"
    g=2
    for dataset in "HASC" "HHAR" "Shoaib" "MotionSense"
    do
        python main.py -g ${g} -version "${version}0" -name ${dataset} --store "${name}0" &
        python main.py -g ${g} -version "${version}1" -name ${dataset} --store "${name}1" &
        python main.py -g ${g} -version "${version}2" -name ${dataset} --store "${name}2" &
        python main.py -g ${g} -version "${version}3" -name ${dataset} --store "${name}3" &
        python main.py -g ${g} -version "${version}4" -name ${dataset} --store "${name}4" &

        wait

        python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name ${dataset} --pretrained "${name}0/${dataset}" --store "${name}0" &
        python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name ${dataset} --pretrained "${name}1/${dataset}" --store "${name}1" &
        python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name ${dataset} --pretrained "${name}2/${dataset}" --store "${name}2" &
        python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name ${dataset} --pretrained "${name}3/${dataset}" --store "${name}3" &
        python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name ${dataset} --pretrained "${name}4/${dataset}" --store "${name}4" 

        wait
    done
done