#!/bin/bash
name='CM_v2_ep'
for e in 2000
do
    python main.py -g 0 -e ${e} -m 'CM' -name HASC --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -m 'CM' -name HHAR --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -m 'CM' -name MotionSense --store "${name}_${e}" &
    python main.py -g 0 -e ${e} -m 'CM' -name Shoaib --store "${name}_${e}" 

    wait
done
