#!/bin/bash
name='CLHAR_revised_v'

for v in 0 1 2 3 4
do
    # python main.py -g 1 --store "${name}_${v}" -version "shot${v}" -name HASC &
    # python main.py -g 1 --store "${name}_${v}" -version "shot${v}" -name HHAR &
    # python main.py -g 1 --store "${name}_${v}" -version "shot${v}" -name MotionSense &
    # python main.py -g 1 --store "${name}_${v}" -version "shot${v}" -name Shoaib 

    # wait

    python transfer.py -g 2 -ft True -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}/HASC" --store "${name}_f1_${v}" &
    python transfer.py -g 2 -ft True -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}/HHAR" --store "${name}_f1_${v}" &
    python transfer.py -g 2 -ft True -version "shot${v}" -shot 10 -name MotionSense --pretrained "${name}_${v}/MotionSense" --store "${name}_f1_${v}" &
    python transfer.py -g 2 -ft True -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}/Shoaib" --store "${name}_f1_${v}" 

    wait

done

# python main.py -g 2 -name HASC --store "${name}_HASC" 
# python main.py -g 2 -name HHAR --store "${name}_HHAR" 
# python main.py -g 2 -name MotionSense --store "${name}_MotionSense" 
# python main.py -g 2 -name Shoaib --store "${name}_Shoaib" 

# python transfer.py -g 3 -version shot -shot 10 -name HASC --pretrained "${name}_HASC" &
# python transfer.py -g 3 -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" &
# python transfer.py -g 3 -version shot -shot 10 -name MotionSense --pretrained "${name}_MotionSense" &
# python transfer.py -g 3 -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib"



# python transfer.py -g 3 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HASC" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HHAR" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_MotionSense" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 3 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HASC" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_MotionSense" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 3 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HASC" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HHAR" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_MotionSense" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 3 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HASC" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HHAR" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_MotionSense" &
# python transfer.py -g 3 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" &