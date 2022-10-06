#!/bin/bash


# for b in 500 1500 2000
# do
#   name="cos_epoch_${b}"
#   python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
# done

name="contrast_w"
python results_analysis.py -name "${name}_HASC_shot" "${name}_HHAR_shot" "${name}_MotionSense_shot" "${name}_Shoaib_shot"