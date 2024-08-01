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

# for a in 45
# do
#     version="alpha${a}_shot"
#     store="mixup_alpha${a}_"
#     for shot in 10
#     do
#         for dataset in "HHAR" 
#         do
            # python main.py -g 3 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
            # python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
            # python main.py -g 3 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
            # python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
            # python main.py -g 3 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#             wait
#         done
#     done
# done

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


# version="users_positions_shot"
# store="mixup_users_positions_"
# for shot in 100
# do
#     for dataset in "Shoaib" 
#     do
#         python main.py -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done


# version="users_devices_shot"
# store="mixup_users_devices_"
# for shot in 100
# do
#     for dataset in "HASC" 
#     do
#         python main.py -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done


# shot=50
# for portion in 60 80 100
# do
#     version="users_positions_tune_portion_${portion}_shot"
#     store="mixup_users_positions_tune_portion_${portion}_"
#     for dataset in "Shoaib" 
#     do
#         python main.py -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done


# for portion in 80
# do
#     version="users_devices_tune_portion_${portion}_shot"
#     store="mixup_users_devices_tune_portion_${portion}_"
#     for dataset in "HASC" 
#     do
        # python main.py -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        # python main.py -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        # python main.py -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        # python main.py -g 0 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 0 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done


# for v in 0 1 2 3 4
# do
# mkdir "runs/mixup_users_devices_tune_portion_40_${v}"
# cp -r "runs/mixup_users_devices_${v}/HASC_shot_50/" "runs/mixup_users_devices_tune_portion_40_${v}"
# mkdir "runs/mixup_users_positions_tune_portion_40_${v}"
# cp -r "runs/mixup_users_positions_${v}/Shoaib_shot_50/" "runs/mixup_users_positions_tune_portion_40_${v}"
# done



# shot=10
# for portion in 45 65
# do
#     version="cd_alpha${portion}_shot"
#     store="mixup_cd_alpha${portion}_shot"
#     for dataset in "HASC" 
#     do
#         # python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         # python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         # python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         # python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done

# for portion in 45 65
# do
#     version="users_devices_alpha${portion}_shot"
#     store="mixup_users_devices_alpha${portion}_shot"
#     for dataset in "HASC" 
#     do
#         python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done

# for portion in 60
# do
#     version="cp_alpha${portion}_shot"
#     store="mixup_cp_alpha${portion}_shot"
#     for dataset in "Shoaib" 
#     do
#         python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done


# for portion in 45 65
# do
#     version="users_positions_alpha${portion}_shot"
#     store="mixup_users_positions_alpha${portion}_shot"
#     for dataset in "Shoaib" 
#     do
#         python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
#         wait
#     done
# done

shot=50
store="mixup_cross_datasets"

# for shot in 5 10 100
# do
#     for dataset in "Merged_dataset"
#     do
#         python main.py -g 3 -version "HASC" --store "${store}/HASC" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "HHAR" --store "${store}/HHAR" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "MotionSense" --store "${store}/MotionSense" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "Shoaib" --store "${store}/Shoaib" -shot ${shot} -name ${dataset}
#         wait
#     done
# done

# for portion in 70 100
# do
#     for dataset in "Merged_dataset"
#     do
#         python main.py -g 3 -version "tune_portion_${portion}_HASC" --store "${store}/HASC" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "tune_portion_${portion}_HHAR" --store "${store}/HHAR" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "tune_portion_${portion}_MotionSense" --store "${store}/MotionSense" -shot ${shot} -name ${dataset} &
#         python main.py -g 3 -version "tune_portion_${portion}_Shoaib" --store "${store}/Shoaib" -shot ${shot} -name ${dataset}
#         wait
#     done
# done

version="leave_shot"
store="mixup_alpha99_"
for shot in 10
do
    for dataset in "Shoaib" 
    do
        python main.py -g 2 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        python main.py -g 2 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        python main.py -g 2 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        python main.py -g 2 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
        python main.py -g 2 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset}
        wait
    done
done