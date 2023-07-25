# g=2
# for v in 0 1 2 3 4
# do

#     store="Mixup_s${v}"
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'HASC' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'HHAR' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'MotionSense' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'Shoaib'

#     wait

# done

# g=3
# name="mixup_cp_adpt"
# version="cp"

# python main.py -g ${g} -version "${version}0" -shot 10 -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name Shoaib --store "${name}4"

# wait

# g=3
# name="mixup_cd_adpt"
# version="cd"

# python main.py -g ${g} -version "${version}0" -shot 10 -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name HASC --store "${name}4"

# wait

# g=3
# name="mixup_cu_adpt"
# version="shot"

# python main.py -g ${g} -version "${version}0" -shot 10 -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name HASC --store "${name}4"


# version="shot"
# shot=50
# for portion in 60 80 100
# do
#     version="tune_portion_${portion}_shot"
#     for v in 0 1 2 3 4
#     do
#         store="mixup_cu_adpt_tune_portion_${portion}_${v}"
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HHAR' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'MotionSense' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'Shoaib'
#         wait

#     done
# done

version="shot"
store="mixup_cu"
for shot in 1
do
    for dataset in "HASC" "HHAR" "MotionSense" "Shoaib" 
    do
        python main.py -g 3 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        python main.py -g 3 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        python main.py -g 3 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        python main.py -g 3 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
        python main.py -g 3 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
    done
    wait
done

# for portion in 60 80 100
# do
#     version="cd_tune_portion_${portion}_shot"
#     store="mixup_cd_tune_portion_${portion}_shot"
#     for shot in 50
#     do
#         python main.py -g 3 -version "${version}0" --store "${store}0" -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}1" --store "${store}1" -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}2" --store "${store}2" -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}3" --store "${store}3" -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}4" --store "${store}4" -shot ${shot} -name 'HASC'
#         wait
#     done
# done


# for portion in 100
# do
#     version="cp_tune_portion_${portion}_shot"
#     store="mixup_cp_tune_portion_${portion}_shot"
#     for shot in 5 10 50 100
#     do
#         python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name 'Shoaib' &
#         python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name 'Shoaib' &
#         python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name 'Shoaib' &
#         python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name 'Shoaib' &
#         python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name 'Shoaib'
#         wait
#     done
# done

# version="train25_supervised_cross"
# store="preliminary_cross"
# python main.py -g 2 -version "${version}" --store "${store}" -shot 5 -name 'HHAR' &
# python main.py -g 2 -version "${version}" --store "${store}" -shot 10 -name 'HHAR' &
# python main.py -g 2 -version "${version}" --store "${store}" -shot 50 -name 'HHAR' &
# python main.py -g 2 -version "${version}" --store "${store}" -shot 100 -name 'HHAR' &
# python main.py -g 2 -version "${version}" --store "${store}" -shot 0 -name 'HHAR'

# version='shot'
# lr=0.0001
<<<<<<< Updated upstream

# for e in 600 800
# do
#     store="mixup_EMG_lr${lr}_e${e}_v"

#     for dataset in "Myo" "NinaPro"
#     do
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}0" --store "${store}0" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}1" --store "${store}1" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}2" --store "${store}2" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}3" --store "${store}3" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}4" --store "${store}4" -shot 10 -name ${dataset} 
=======

# for e in 600 800
# do
#     store="mixup_EMG_lr${lr}_e${e}_v"

#     for dataset in "Myo" "NinaPro"
#     do
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}0" --store "${store}0" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}1" --store "${store}1" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}2" --store "${store}2" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}3" --store "${store}3" -shot 10 -name ${dataset} &
#         python main.py -lr ${lr} -e ${e} -g 3 -version "${version}4" --store "${store}4" -shot 10 -name ${dataset} 

#         wait
#     done

# done


version='shot'
lr=0.005
e=200

for shot in 1 5 10 50
do
    store="mixup_EMG_"

    for dataset in "Myo" "NinaPro"
    do
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset} 
>>>>>>> Stashed changes

#         wait
#     done

<<<<<<< Updated upstream
# done
=======
done

shot=50
for portion in 60 80 100
do
    version="cu_tune_portion_${portion}_shot"
    store="mixup_EMG_portion_${portion}_"

    for dataset in "Myo" "NinaPro"
    do
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
        python main.py -lr ${lr} -e ${e} -g 1 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset} 

        wait
    done
done
>>>>>>> Stashed changes
