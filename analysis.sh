#!/bin/bash
name="runs/Origin_w_transfer_DAL_uni_lr1e-4_slr"
# name="baseline/MMD/runs/FM_aug_ep_"
python results_analysis.py -name "${name}0.9_bt32"

