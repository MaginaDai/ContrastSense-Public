#!/bin/bash
key="ewc_pretrain"
name="runs/f1_change_no_"
python results_analysis.py -name "${name}0" "${name}1" "${name}2" "${name}3" "${name}4" 
# python results_analysis.py -name "runs/Shot_Rotate_p1_lam10_ewc_pretrain_2"
# python results_analysis.py -name "${name}0_${key}" "${name}1_${key}" "${name}2_${key}" "${name}3_${key}" "${name}4_${key}" 


# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0
