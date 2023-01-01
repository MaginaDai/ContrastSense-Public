#!/bin/bash
name="Shot"
version="shot"


name="slr"
version="shot"
# v=
# 
# 
for slr in 0.4 0.5
do
    for v in 0 1 2 3 4
    do
        python main.py -g 0 -label_type 1 -slr ${slr} -version "${version}${v}" -name HASC --store "${name}_slr${slr}_${v}" &
        python main.py -g 0 -label_type 1 -slr ${slr} -version "${version}${v}" -name HHAR --store "${name}_slr${slr}_${v}" &
        python main.py -g 1 -label_type 1 -slr ${slr} -version "${version}${v}" -name MotionSense --store "${name}_slr${slr}_${v}" &
        python main.py -g 1 -label_type 1 -slr ${slr} -version "${version}${v}" -name Shoaib --store "${name}_slr${slr}_${v}"

        wait
        
        python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}_slr${slr}_${v}/HASC" --store "${name}_slr${slr}_ewc_${v}" &
        python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}_slr${slr}_${v}/HHAR" --store "${name}_slr${slr}_ewc_${v}" &
        python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}_slr${slr}_${v}/Shoaib" --store "${name}_slr${slr}_ewc_${v}" &
        python main_trans_ewc.py -g 0 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name MotionSense  --pretrained "${name}_slr${slr}_${v}/MotionSense" --store "${name}_slr${slr}_ewc_${v}"

        wait

        # python main_transfer.py -g 2 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}_${v}/HASC" --store "${name}_${v}_wo_ewc" &
        # python main_transfer.py -g 2 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}_${v}/HHAR" --store "${name}_${v}_wo_ewc" &
        # python main_transfer.py -g 3 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}/Shoaib" --store "${name}_${v}_wo_ewc" &
        # python main_transfer.py -g 3 -ft True -lr 0.0005 -version "${version}${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}/MotionSense" --store "${name}_${v}_wo_ewc" & #######

        # wait
    done
done
