#!/bin/bash
# key="wo_ewc"
name="runs/trans_p_1_"
# python results_analysis.py -name "${name}0" "${name}1" "${name}2" "${name}3"  "${name}4"
python results_analysis.py -name "runs/trans_p_1_2_wo_ewc"
# python results_analysis.py -name "${name}0_${key}" "${name}1_${key}" "${name}2_${key}" "${name}3_${key}" "${name}4_${key}" 


# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0
