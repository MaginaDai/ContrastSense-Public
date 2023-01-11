#!/bin/bash
key="v3"
name="baseline/Mixup/runs/Mixup_v3_lr"
# python results_analysis.py -name "${name}0" "${name}1" "${name}2" "${name}3" "${name}4"
# python results_analysis.py -name "runs/plain_Negate_aug0_v1" "runs/plain_Negate_aug0.2_v1" "runs/plain_flip_aug0.2_ftdimfix_v1" "runs/plain_Negate_aug0.8_v1" "runs/plain_Negate_aug1.0_v1" 
# python results_analysis.py -name "${name}50_${key}" "${name}100_${key}" "${name}150_${key}"
python results_analysis.py -name "${name}0.00001" "${name}0.00005" "${name}0.0001" "${name}0.0005" "${name}0.001" "${name}0.005"

# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0

####### preliminary ########
# percent=65
# python results_analysis.py -name "v0_${percent}" "v1_${percent}" "v2_${percent}" "v3_${percent}" "v4_${percent}"