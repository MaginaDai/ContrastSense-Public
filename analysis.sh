#!/bin/bash
key=""
name="runs/trans_SCL_0.01_bcl_512"
# name="baseline/MMD/runs/FM_aug_ep_"
python results_analysis.py -name "runs/ewc_v4_e400_0" "runs/ewc_v4_e400_1" "runs/ewc_v4_e400_2" "runs/ewc_v4_e400_3" "runs/ewc_v4_e400_4"
# python results_analysis.py -name "runs/ewc_v4"
