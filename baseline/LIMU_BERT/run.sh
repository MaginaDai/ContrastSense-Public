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

name=a_shot

for v in 0 1 2 3 4
do

    python pretrain.py v1 "${name}${v}" -g 0 -s MotionSense -name 'MotionSense' &
    python pretrain.py v1 "${name}${v}" -g 0 -s HHAR -name 'HHAR' &
    python pretrain.py v1 "${name}${v}" -g 1 -s HASC -name 'HASC' &
    python pretrain.py v1 "${name}${v}" -g 1 -s Shoaib -name 'Shoaib'

    wait

    python classifier_bert.py v1_v2 "${name}${v}" -g 0 -shot 10 -p MotionSense -f "MotionSense" -name 'MotionSense' -s "limu_gru_MotionSense_shot${v}" &
    python classifier_bert.py v1_v2 "${name}${v}" -g 0 -shot 10 -p HHAR -f "HHAR" -name 'HHAR' -s "limu_gru_HHAR_shot${v}" &
    python classifier_bert.py v1_v2 "${name}${v}" -g 1 -shot 10 -p HASC -f "HASC" -name 'HASC' -s "limu_gru_HASC_shot${v}" &
    python classifier_bert.py v1_v2 "${name}${v}" -g 1 -shot 10 -p Shoaib -f "Shoaib" -name 'Shoaib' -s "limu_gru_Shoaib_shot${v}"

    wait

done
