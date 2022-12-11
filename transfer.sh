#!/bin/bash

name="Origin_w"
store='trans_SCL'

for bcl in 512
do
    for slr in 0.01
    do
        python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HASC --pretrained "${name}_HASC" --store "${store}_${slr}_bcl_${bcl}" &
        python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 0 -ft True -lr 0.0005 -version "shot" -shot 10 -name HHAR --pretrained "${name}_HHAR" --store "${store}_${slr}_bcl_${bcl}" &
        python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name Shoaib --pretrained "${name}_Shoaib" --store "${store}_${slr}_bcl_${bcl}" &
        python main_trans_SCL.py -bcl ${bcl} -cl_slr ${slr} -g 1 -ft True -lr 0.0005 -version "shot" -shot 10 -name MotionSense  --pretrained "${name}_MotionSense" --store "${store}_${slr}_bcl_${bcl}"
        
        wait
    done
done

