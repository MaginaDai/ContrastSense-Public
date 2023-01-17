#!/bin/bash
key="v1"
name="runs/CDL_ewc_aug_no_negate_v"
python results_analysis.py -name "${name}0" "${name}1" "${name}2" "${name}3" "${name}4" 
# python results_analysis.py -name "runs/CDL_ewc_mixup_v0" 
# python results_analysis.py -name "${name}0_${key}" "${name}0.2_${key}" "${name}0.8_${key}" "${name}1.0_${key}"
# python results_analysis.py -name "${name}0.00001" "${name}0.00005" "${name}0.0001" "${name}0.0005" "${name}0.001" "${name}0.005" "${name}0.01" "${name}0.05" "${name}0.1"

# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0

####### preliminary ########
# percent=65
# python results_analysis.py -name "v0_${percent}" "v1_${percent}" "v2_${percent}" "v3_${percent}" "v4_${percent}"