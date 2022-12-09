#!/bin/bash
fm=0.01
# name="runs/ewc_v4_fm${fm}_lam"
name="baseline/MMD/runs/FM_aug_ep_"
python results_analysis.py -name "${name}500" "${name}1000" "${name}1500" "${name}2000"
