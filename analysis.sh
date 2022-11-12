#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

# ad_lr=0.000006 0.000007 0.000008 0.000009 0.00001 0.00002 0.00003 0.00004 0.00005
# name="Origin_wo_transfer_DAL_lr${ad_lr}_sep"

name="runs/Origin_wo_transfer_DAL_uni_slr"
# t=0.5
python results_analysis.py -name "${name}0.0_bt32" "${name}0.1_bt32" "${name}0.2_bt32" "${name}0.3_bt32" "${name}0.4_bt32" "${name}0.5_bt32" "${name}0.6_bt32" "${name}0.7_bt32" "${name}0.8_bt32" "${name}0.9_bt32" "${name}1.0_bt32"

method="ClusterHAR"
head="Rotate_cluster"
name="baseline/${method}/runs/${head}"
# name="runs/Origin_w"

# python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"

# python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
