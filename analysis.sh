#!/bin/bash
fm=0.01
name="runs/ewc_v4_fm${fm}_lam"
python results_analysis.py -name "${name}1000" "${name}2500" "${name}5000" "${name}7500" "${name}10000"
