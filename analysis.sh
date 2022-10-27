#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

# ad_lr=0.000006 0.000007 0.000008 0.000009 0.00001 0.00002 0.00003 0.00004 0.00005
# name="Origin_wo_transfer_DAL_lr${ad_lr}_sep"
name="DAL_t"
t=0.5
python results_analysis.py -name "${name}_t_${t}_HASC" "${name}_t_${t}_HHAR" "${name}_t_${t}_MotionSense" "${name}_t_${t}_Shoaib"