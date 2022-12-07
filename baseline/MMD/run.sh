#!/bin/bash
name='FM_aug_ep'
for e in 150 200 250 300 350 400 450 500
do
    python main.py -g 0 -e ${e} -name HASC --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -name HHAR --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -name MotionSense --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -name Shoaib --store "${name}_${e}" 

    wait
done
