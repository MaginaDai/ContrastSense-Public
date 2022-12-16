#!/bin/bash

for v in 0 1 2 3 4
do
    name='CM_v'
    python main.py -g 3 -m 'CM' -name HASC --store "${name}_${v}" &
    python main.py -g 3 -m 'CM' -name HHAR --store "${name}_${v}" &
    python main.py -g 3 -m 'CM' -name MotionSense --store "${name}_${v}" &
    python main.py -g 3 -m 'CM' -name Shoaib --store "${name}_${v}" 

    wait

    name='FM_v'
    python main.py -g 3 -m 'FM' -name HASC --store "${name}_${v}" &
    python main.py -g 3 -m 'FM' -name HHAR --store "${name}_${v}" &
    python main.py -g 3 -m 'FM' -name MotionSense --store "${name}_${v}" &
    python main.py -g 3 -m 'FM' -name Shoaib --store "${name}_${v}" 

    wait

done
