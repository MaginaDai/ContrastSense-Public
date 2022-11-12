#!/bin/bash
name="Rotate_cluster"

python main.py -g 0 -do_cluster True -name HASC --store "${name}_HASC" &
python main.py -g 0 -do_cluster True -name HHAR --store "${name}_HHAR" &
python main.py -g 1 -do_cluster True -name MotionSense --store "${name}_MotionSense" &
python main.py -g 1 -do_cluster True -name Shoaib --store "${name}_Shoaib"

wait

# python transfer.py -g 0 -shot 10 -version 'shot' -name HASC --pretrained "HASC" &
# python transfer.py -g 0 -shot 10 -version 'shot' -name HHAR --pretrained "HHAR" &
# python transfer.py -g 1 -shot 10 -version 'shot' -name MotionSense --pretrained "MotionSense" &
# python transfer.py -g 1 -shot 10 -version 'shot' -name Shoaib --pretrained "Shoaib" 

python transfer.py -g 0 -lr 0.01 -shot 10 -version 'shot' -name HASC --pretrained "${name}_HASC" &
python transfer.py -g 0 -lr 0.01 -shot 10 -version 'shot' -name HHAR --pretrained "${name}_HHAR" &
python transfer.py -g 1 -lr 0.01 -shot 10 -version 'shot' -name MotionSense --pretrained "${name}_MotionSense" &
python transfer.py -g 1 -lr 0.01 -shot 10 -version 'shot' -name Shoaib --pretrained "${name}_Shoaib"


# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_MotionSense"
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" &