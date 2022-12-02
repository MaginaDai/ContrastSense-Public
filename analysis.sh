#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

# ad_lr=0.000006 0.000007 0.000008 0.000009 0.00001 0.00002 0.00003 0.00004 0.00005
# name="Origin_wo_transfer_DAL_lr${ad_lr}_sep"
lr=0.0001
name="runs/Origin_w_ewc_lr${lr}_e"
# t=0.5
python results_analysis.py -name "${name}100" "${name}200" "${name}300" "${name}500"
# python results_analysis.py -name "runs/Origin_w_ewc_fm0.01_wd0"

# method="ClusterHAR"
# head="Rotate_cluster"
# name="baseline/${method}/runs/${head}"
# name="runs/Origin_w"

# python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"

# python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
