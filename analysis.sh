#!/bin/bash
key="v2"
name="runs/f1_change_no_"
python results_analysis.py -name "${name}0" "${name}1" "${name}2" "${name}3" "${name}4"
# python results_analysis.py -name "runs/f1_change_no_2"
# python results_analysis.py -name "${name}256_${key}" "${name}512_${key}" "${name}1024_${key}" "${name}2048_${key}" "${name}3072_${key}" 


# python results_analysis.py -name baseline/GILE/runs/GILE_a_t25_shot0_v_0
