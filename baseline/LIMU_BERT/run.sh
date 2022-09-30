#!/bin/bash
# nohup python -u pretrain.py v1 motion 20_120 -g 0 -s motion > log/pretrain_base_motion.log &
# nohup python -u pretrain.py v1 uci 20_120 -g 0 -s uci > log/pretrain_base_uci.log &
# nohup python -u pretrain.py v1 hhar 20_120 -g 0 -s hhar > log/pretrain_base_hhar.log &
# nohup python -u pretrain.py v1 shoaib 20_120 -g 0 -s shoaib > log/pretrain_base_shoaib.log &

# python -u pretrain.py v1 20_120 -g 0 -s motion -name 'HHAR person'

# python classifier_bert.py v1_v2 20_120 -g 1 -s motion -name 'HHAR person' -f motion -s limu_gru_per05 -percent 0.5

# python -u pretrain.py v1 20_120 -g 0 -s MotionSense -name 'MotionSense' &
# python -u pretrain.py v1 20_120 -g 0 -s Shoaib -name 'Shoaib' &
# python -u pretrain.py v1 20_120 -g 0 -s UCI -name 'UCI'
# python -u pretrain.py v1 50_200 -g 0 -s HHAR_50_200 -name 'HHAR 50_200'

# wait

# python classifier_bert.py v1_v2 20_120 -p HHAR -f CrossVal_HHAR -name 'UCI' -s limu_gru_UCI_1 &
# python classifier_bert.py v1_v2 20_120 -p HHAR -f CrossVal_HHAR -name 'MotionSense' -s limu_gru_MotionSense_1 &
# python classifier_bert.py v1_v2 20_120 -p HHAR -f CrossVal_HHAR -name 'Shoaib' -s limu_gru_Shoaib_1 &
# python classifier_bert.py v1_v2 50_200 -p 'HHAR 50_200' -f HHAR_50_200 -name 'HHAR 50_200' -s limu_gru_hhar_50_200


# wait

# python classifier_bert.py v1_v2 20_120 -f MotionSense -name 'UCI' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 20_120 -f Shoaib -name 'UCI' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 20_120 -f HHAR -name 'UCI' -s limu_gru_hhar &
# python classifier_bert.py v1_v2 20_120 -f UCI -name 'UCI' -s limu_gru_UCI 

# wait

# python classifier_bert.py v1_v2 20_120 -f MotionSense -name 'Shoaib' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 20_120 -f Shoaib -name 'Shoaib' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 20_120 -f HHAR -name 'Shoaib' -s limu_gru_hhar &
# python classifier_bert.py v1_v2 20_120 -f UCI -name 'Shoaib' -s limu_gru_UCI


# wait

# python classifier_bert.py v1_v2 20_120 -f MotionSense -name 'MotionSense' -s limu_gru_MotionSense &
# python classifier_bert.py v1_v2 20_120 -f Shoaib -name 'MotionSense' -s limu_gru_Shoaib &
# python classifier_bert.py v1_v2 20_120 -f HHAR -name 'MotionSense' -s limu_gru_hhar &
# python classifier_bert.py v1_v2 20_120 -f UCI -name 'MotionSense' -s limu_gru_UCI &


# for num in 2 3 4 5 6
# do
#     python -u pretrain.py v1 "50_200_test_${num}" -g 0 -s "HHAR_50_200_test_${num}" -name 'HHAR'
#     python classifier_bert.py v1_v2 "50_200_test_${num}" -p "HHAR" -f "HHAR_50_200_test_${num}" -name 'HHAR' -s "limu_gru_HHAR_50_200_test_${num}"
# done

name='50_200'
dataset='ICHAR'
# python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"

# dataset='Shoaib'
# python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
# python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"

# for dataset in 'HHAR' 'MotionSense' 'shoaib' 'UCI' 'ICHAR' 'HASC'
# do
#     python -u pretrain.py v1 ${name} -name ${dataset} -g 0 -s "${dataset}_${name}" 
#     python classifier_bert.py v1_v2 ${name} -p ${dataset} -f "${dataset}_${name}" -name ${dataset} -s "${dataset}_${name}"
# done
