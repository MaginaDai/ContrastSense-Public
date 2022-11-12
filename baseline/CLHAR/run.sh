#!/bin/bash
name='Ori'
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