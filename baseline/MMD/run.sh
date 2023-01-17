#!/bin/bash

for v in 0 1 2 3 4
do

    name='CM_s'
    python main.py -g 2 -version "s${v}" -m 'CM' -name HASC --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'CM' -name HHAR --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'CM' -name MotionSense --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'CM' -name Shoaib --store "${name}_${v}" 

    wait

    name='FM_s'
    python main.py -g 2 -version "s${v}" -m 'FM' -name HASC --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'FM' -name HHAR --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'FM' -name MotionSense --store "${name}_${v}" &
    python main.py -g 2 -version "s${v}" -m 'FM' -name Shoaib --store "${name}_${v}" 

    wait

done
