#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

slr=0.00005
name="DAL_lr${slr}"
python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"