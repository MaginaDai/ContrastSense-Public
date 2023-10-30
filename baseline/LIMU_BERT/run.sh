#!/bin/bash
# nohup python -u pretrain.py v1 motion 20_120 -g 0 -s motion > log/pretrain_base_motion.log &
# nohup python -u pretrain.py v1 uci 20_120 -g 0 -s uci > log/pretrain_base_uci.log &
# nohup python -u pretrain.py v1 hhar 20_120 -g 0 -s hhar > log/pretrain_base_hhar.log &
# nohup python -u pretrain.py v1 shoaib 20_120 -g 0 -s shoaib > log/pretrain_base_shoaib.log &

# python -u pretrain.py v1 20_120 -g 0 -s motion -name 'HHAR person'

# python classifier_bert.py v1_v2 20_120 -g 1 -s motion -name 'HHAR person' -f motion -s limu_gru_per05 -percent 0.5

# name='shot'

# python -u pretrain.py v1 ${name} -g 0 -s MotionSense -name 'MotionSense' &
# python -u pretrain.py v1 ${name} -g 0 -s Shoaib -name 'Shoaib'

# wait

# python -u pretrain.py v1 ${name} -g 0 -s HASC -name 'HASC' &
# python -u pretrain.py v1 ${name} -g 0 -s HHAR -name 'HHAR'

# wait

# dataset='ICHAR'
# python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
# python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"

# python classifier_bert.py v1_v2 ${name} -shot 10 -p MotionSense -f "MotionSense" -name 'MotionSense' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p Shoaib -f "Shoaib" -name 'MotionSense' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HHAR -f "HHAR" -name 'MotionSense' -s limu_gru_HHAR &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HASC -f "HASC" -name 'MotionSense' -s limu_gru_HASC

# wait

# python classifier_bert.py v1_v2 ${name} -shot 10 -p MotionSense -f "MotionSense" -name 'HHAR' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p Shoaib -f "Shoaib" -name 'HHAR' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HHAR -f "HHAR" -name 'HHAR' -s limu_gru_HHAR &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HASC -f "HASC" -name 'HHAR' -s limu_gru_HASC

# wait

# python classifier_bert.py v1_v2 ${name} -shot 10 -p MotionSense -f "MotionSense" -name 'HASC' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p Shoaib -f "Shoaib" -name 'HASC' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HHAR -f "HHAR" -name 'HASC' -s limu_gru_HHAR
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HASC -f "HASC" -name 'HASC' -s limu_gru_HASC

# wait

# python classifier_bert.py v1_v2 ${name} -shot 10 -p MotionSense -f "MotionSense" -name 'Shoaib' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p Shoaib -f "Shoaib" -name 'Shoaib' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HHAR -f "HHAR" -name 'Shoaib' -s limu_gru_HHAR &
# python classifier_bert.py v1_v2 ${name} -shot 10 -p HASC -f "HASC" -name 'Shoaib' -s limu_gru_HASC





# for num in 2 3 4 5 6
# do
#     python -u pretrain.py v1 "50_200_test_${num}" -g 0 -s "HHAR_50_200_test_${num}" -name 'HHAR'
#     python classifier_bert.py v1_v2 "50_200_test_${num}" -p "HHAR" -f "HHAR_50_200_test_${num}" -name 'HHAR' -s "limu_gru_HHAR_50_200_test_${num}"
# done



# dataset='Shoaib'
# python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
# python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"

# for dataset in 'HHAR' 'MotionSense' 'shoaib' 'UCI' 'ICHAR' 'HASC'
# do
#     python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
#     python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"
# done

# name="s"
# for v in 0 1 2 3 4
# do

#     python pretrain.py v1 "${name}${v}" -g 0 -f "${name}" -s MotionSense -name 'MotionSense' &
#     python pretrain.py v1 "${name}${v}" -g 0 -f "${name}" -s HHAR -name 'HHAR' &
#     python pretrain.py v1 "${name}${v}" -g 1 -f "${name}" -s HASC -name 'HASC' &
#     python pretrain.py v1 "${name}${v}" -g 1 -f "${name}" -s Shoaib -name 'Shoaib'

#     wait

#     python classifier_bert.py v1_v2 "${name}${v}" -g 0 -f "${name}" -shot 60 -p MotionSense -name 'MotionSense' -s "limu_gru_MotionSense_shot${v}" &
#     python classifier_bert.py v1_v2 "${name}${v}" -g 0 -f "${name}" -shot 60 -p HHAR -name 'HHAR' -s "limu_gru_HHAR_shot${v}" &
#     python classifier_bert.py v1_v2 "${name}${v}" -g 1 -f "${name}" -shot 60 -p HASC -name 'HASC' -s "limu_gru_HASC_shot${v}" &
#     python classifier_bert.py v1_v2 "${name}${v}" -g 1 -f "${name}" -shot 60 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_shot${v}"

#     wait

# done


# version="cp"
# name="position"

# python pretrain.py v1 "${version}0" -g 0 -f "${name}" -s Shoaib -name 'Shoaib' &
# python pretrain.py v1 "${version}1" -g 0 -f "${name}" -s Shoaib -name 'Shoaib' &
# python pretrain.py v1 "${version}2" -g 1 -f "${name}" -s Shoaib -name 'Shoaib' &
# python pretrain.py v1 "${version}3" -g 1 -f "${name}" -s Shoaib -name 'Shoaib' &
# python pretrain.py v1 "${version}4" -g 1 -f "${name}" -s Shoaib -name 'Shoaib'

# wait

# python classifier_bert.py v1_v2 "${version}0" -g 0 -f "${name}" -shot 10 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_${name}0" &
# python classifier_bert.py v1_v2 "${version}1" -g 0 -f "${name}" -shot 10 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_${name}1" &
# python classifier_bert.py v1_v2 "${version}2" -g 1 -f "${name}" -shot 10 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_${name}2" &
# python classifier_bert.py v1_v2 "${version}3" -g 1 -f "${name}" -shot 10 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_${name}3" &
# python classifier_bert.py v1_v2 "${version}4" -g 1 -f "${name}" -shot 10 -p Shoaib -name 'Shoaib' -s "limu_gru_Shoaib_${name}4"

# wait

# version="shot"
# name="HASC_users"

# python pretrain.py v1 "${version}0" -g 0 -f "${name}" -s HASC -name 'HASC' &
# python pretrain.py v1 "${version}1" -g 0 -f "${name}" -s HASC -name 'HASC' &
# python pretrain.py v1 "${version}2" -g 2 -f "${name}" -s HASC -name 'HASC' &
# python pretrain.py v1 "${version}3" -g 2 -f "${name}" -s HASC -name 'HASC' &
# python pretrain.py v1 "${version}4" -g 2 -f "${name}" -s HASC -name 'HASC'

# wait

# python classifier_bert.py v1_v2 "${version}0" -g 0 -f "${name}" -shot 10 -p HASC -name 'HASC' -s "HASC_ft_shot_10" &
# python classifier_bert.py v1_v2 "${version}1" -g 0 -f "${name}" -shot 10 -p HASC -name 'HASC' -s "HASC_ft_shot_10" &
# python classifier_bert.py v1_v2 "${version}2" -g 2 -f "${name}" -shot 10 -p HASC -name 'HASC' -s "HASC_ft_shot_10"
# python classifier_bert.py v1_v2 "${version}3" -g 2 -f "${name}" -shot 10 -p HASC -name 'HASC' -s "HASC_ft_shot_10" &
# python classifier_bert.py v1_v2 "${version}4" -g 2 -f "${name}" -shot 10 -p HASC -name 'HASC' -s "HASC_ft_shot_10"

# wait

# version="shot"
# name="user"
# shot=50
# for portion in 60 80 100
# do 
#     pretrain_version="shot"
#     version="tune_portion_${portion}_shot"
#     store="portion${portion}_ft_shot_${shot}"
#     for v in 0 1 2 3 4
#     do
#         python classifier_bert.py v1_v2 "${version}${v}" -g 2 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p MotionSense -name 'MotionSense' -s "MotionSense_${store}" &
#         python classifier_bert.py v1_v2 "${version}${v}" -g 2 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}"
#         wait

#         python classifier_bert.py v1_v2 "${version}${v}" -g 2 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}${v}" -g 2 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"

#         wait

#     done
# done

# version="shot"
# name="user"
# shot=1
# for portion in 40
# do 
#     pretrain_version="shot"
#     version="shot"
#     store="ft_shot_${shot}"
    # for v in 0 1 2 3 4 
    # do
    #     python classifier_bert.py v1_v2 "${version}${v}" -g 0 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p MotionSense -name 'MotionSense' -s "MotionSense_${store}" &
    #     python classifier_bert.py v1_v2 "${version}${v}" -g 0 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
    #     python classifier_bert.py v1_v2 "${version}${v}" -g 0 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
    #     python classifier_bert.py v1_v2 "${version}${v}" -g 0 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"

    #     wait

    # done

#     python classifier_bert.py v1_v2 "${version}3" -g 0 -f "${name}" -pv "${pretrain_version}3" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
#     python classifier_bert.py v1_v2 "${version}4" -g 0 -f "${name}" -pv "${pretrain_version}4" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# done

# version="cp"
# name="positions"
# for shot in 5 50 100
# do 
#     pretrain_version="cp"
#     version="cp"
#     store="ft_shot_${shot}"
#     for v in 0 1 2 3 4
#     do
#         python classifier_bert.py v1_v2 "${version}${v}" -g 1 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p MotionSense -name 'MotionSense' -s "MotionSense_${store}" &
#         python classifier_bert.py v1_v2 "${version}${v}" -g 1 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}"
        
#         wait

#         python classifier_bert.py v1_v2 "${version}${v}" -g 1 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}${v}" -g 1 -f "${name}" -pv "${pretrain_version}${v}" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"

#         wait

#     done
# done

# version="cp"
# name="position"
# for shot in 5 50 100
# do 
#     pretrain_version="cp"
#     version="cp"
#     store="ft_shot_${shot}"
#     python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -pv "${pretrain_version}0" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -pv "${pretrain_version}1" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}2" -g 2 -f "${name}" -pv "${pretrain_version}2" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"
    
#     wait

#     python classifier_bert.py v1_v2 "${version}3" -g 2 -f "${name}" -pv "${pretrain_version}3" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}4" -g 2 -f "${name}" -pv "${pretrain_version}4" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"

#     wait
# done

# version="cd"
# name="devices"
# pretrain_version="cd"
# for shot in 5 50 100
# do 
#     store="ft_shot_${shot}"
#     python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -pv "${pretrain_version}0" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#     python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -pv "${pretrain_version}1" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#     python classifier_bert.py v1_v2 "${version}2" -g 2 -f "${name}" -pv "${pretrain_version}2" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}"
    
#     wait

#     python classifier_bert.py v1_v2 "${version}3" -g 2 -f "${name}" -pv "${pretrain_version}3" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#     python classifier_bert.py v1_v2 "${version}4" -g 2 -f "${name}" -pv "${pretrain_version}4" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}"

#     wait
# done

# for portion in 60 80 100
# do
# version="cd_tune_portion_${portion}_shot"
# name="devices"
# pretrain_version="cd"
#     for shot in 50
#     do 
#         store="ft_shot_${shot}"
#         python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -pv "${pretrain_version}0" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -pv "${pretrain_version}1" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}2" -g 2 -f "${name}" -pv "${pretrain_version}2" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}3" -g 3 -f "${name}" -pv "${pretrain_version}3" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}" &
#         python classifier_bert.py v1_v2 "${version}4" -g 3 -f "${name}" -pv "${pretrain_version}4" -shot ${shot} -p HASC -name 'HASC' -s "HASC_${store}"

#         wait
#     done
# done

# version="cp_tune_portion_100_shot"
# name="position"
# pretrain_version="cp"
# for shot in 5 10 50 100
# do 
#     store="ft_shot_${shot}"
#     python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -pv "${pretrain_version}0" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -pv "${pretrain_version}1" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}2" -g 2 -f "${name}" -pv "${pretrain_version}2" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}3" -g 3 -f "${name}" -pv "${pretrain_version}3" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}" &
#     python classifier_bert.py v1_v2 "${version}4" -g 3 -f "${name}" -pv "${pretrain_version}4" -shot ${shot} -p Shoaib -name 'Shoaib' -s "Shoaib_${store}"

#     wait
# done

# name='preliminary'

# python pretrain.py v1 "train25_supervised_random" -g 0 -f "${name}" -s HHAR -name 'HHAR' &
# python pretrain.py v1 "train45_supervised_random" -g 0 -f "${name}" -s HHAR -name 'HHAR' &
# python pretrain.py v1 "train65_supervised_random" -g 0 -f "${name}" -s HHAR -name 'HHAR' &
# python pretrain.py v1 "train25_supervised_label" -g 1 -f "${name}" -s HHAR -name 'HHAR' &
# python pretrain.py v1 "train45_supervised_label" -g 1 -f "${name}" -s HHAR -name 'HHAR' &
# python pretrain.py v1 "train65_supervised_label" -g 1 -f "${name}" -s HHAR -name 'HHAR'

# wait
# shot=50
# store="ft_shot_${shot}"

# python classifier_bert.py v1_v2 "train25_supervised_random" -g 0 -f "${name}" -pv "train25_supervised_random" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# python classifier_bert.py v1_v2 "train45_supervised_random" -g 0 -f "${name}" -pv "train45_supervised_random" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# python classifier_bert.py v1_v2 "train65_supervised_random" -g 0 -f "${name}" -pv "train65_supervised_random" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# python classifier_bert.py v1_v2 "train25_supervised_label" -g 1 -f "${name}" -pv "train25_supervised_label" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# python classifier_bert.py v1_v2 "train45_supervised_label" -g 1 -f "${name}" -pv "train45_supervised_label" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}" &
# python classifier_bert.py v1_v2 "train65_supervised_label" -g 1 -f "${name}" -pv "train65_supervised_label" -shot ${shot} -p HHAR -name 'HHAR' -s "HHAR_${store}"



version="users_positions_shot"
name="multiple"
for dataset in "Shoaib"
do
    
    python pretrain.py v1 "${version}0" -g 2 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}1" -g 2 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}2" -g 3 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}3" -g 3 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}4" -g 3 -f "${name}" -s ${dataset} -name "${dataset}"

    wait
done

dataset="Shoaib"
for shot in 1 5 10 20 50
do
    python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -shot ${shot} -pv "${version}0" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -shot ${shot} -pv "${version}1" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}2" -g 3 -f "${name}" -shot ${shot} -pv "${version}2" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}3" -g 3 -f "${name}" -shot ${shot} -pv "${version}3" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}4" -g 3 -f "${name}" -shot ${shot} -pv "${version}4" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}"

    wait
done


version="users_devices_shot"
name="multiple"
for dataset in "HASC"
do
    
    python pretrain.py v1 "${version}0" -g 2 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}1" -g 2 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}2" -g 3 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}3" -g 3 -f "${name}" -s ${dataset} -name "${dataset}" &
    python pretrain.py v1 "${version}4" -g 3 -f "${name}" -s ${dataset} -name "${dataset}"

    wait
done

dataset="HASC"
for shot in 1 5 10 20 50
do
    python classifier_bert.py v1_v2 "${version}0" -g 2 -f "${name}" -shot ${shot} -pv "${version}0" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}1" -g 2 -f "${name}" -shot ${shot} -pv "${version}1" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}2" -g 3 -f "${name}" -shot ${shot} -pv "${version}2" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}3" -g 3 -f "${name}" -shot ${shot} -pv "${version}3" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}" &
    python classifier_bert.py v1_v2 "${version}4" -g 3 -f "${name}" -shot ${shot} -pv "${version}4" -p "${dataset}" -name "${dataset}" -s "${dataset}_ft_shot_${shot}"

    wait
done