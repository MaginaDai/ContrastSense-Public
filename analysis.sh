#!/bin/bash
key="v1"
lr=0.001
name="runs/improve_v1_ewc_w1000_"
# name="baseline/CALDA/runs/CALDA_w1_v"
python results_analysis.py -shot 10 -name "${name}0" "${name}1" "${name}2" "${name}3" "${name}4"
# python results_analysis.py -name "runs/CDL_ewc_mixup_v0" 
# python results_analysis.py -name "${name}0_${key}" "${name}0.2_${key}" "${name}0.8_${key}" "${name}1.0_${key}"
# python results_analysis.py -name "${name}0.1" "${name}0.2" "${name}0.3" "${name}0.4" "${name}0.5" "${name}0.6" "${name}0.7" "${name}0.8" "${name}0.9" "${name}1.0"

# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0

####### preliminary ########
# percent=65
# python results_analysis.py -name "v0_${percent}" "v1_${percent}" "v2_${percent}" "v3_${percent}" "v4_${percent}"