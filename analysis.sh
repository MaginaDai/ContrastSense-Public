#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

lr=0.00001
name="DAL_w_normalize"
python results_analysis.py -name "Origin_wo_transfer_DAL_lr${lr}_sep"