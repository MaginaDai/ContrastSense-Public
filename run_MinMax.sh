#!/bin/bash
name="MinMax_v1"
main_lr=0.0001
main_lr_name="0_0005"
lr=0.0005

python main_MinMax.py --store "${name}_e_1000" -lr-selector ${lr} -e 1000
python main_transfer.py --pretrained "./runs/${name}_e_1000/model_best.pth.tar" --store "${name}_lr_${lr}" -if-fine-tune True -e 300
