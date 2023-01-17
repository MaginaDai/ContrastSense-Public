name='CPC_s'

for v in 0 1 2 3 4
do
    python main.py -g 1 --store "${name}${v}" -version "s${v}" -name HASC &
    python main.py -g 1 --store "${name}${v}" -version "s${v}" -name HHAR &
    python main.py -g 1 --store "${name}${v}" -version "s${v}" -name MotionSense &
    python main.py -g 1 --store "${name}${v}" -version "s${v}" -name Shoaib 

    wait

    python transfer.py -g 1 -ft True -version "s${v}" -shot 60 -name HASC --pretrained "${name}${v}/HASC" --store "${name}${v}" &
    python transfer.py -g 1 -ft True -version "s${v}" -shot 60 -name HHAR --pretrained "${name}${v}/HHAR" --store "${name}${v}" &
    python transfer.py -g 1 -ft True -version "s${v}" -shot 60 -name MotionSense --pretrained "${name}${v}/MotionSense" --store "${name}${v}" &
    python transfer.py -g 1 -ft True -version "s${v}" -shot 60 -name Shoaib --pretrained "${name}${v}/Shoaib" --store "${name}${v}" 

    wait

done

# python main.py -g 3 -name HHAR -version "shot_portion35" --store "${name}_HHAR_portion35" &
# python main.py -g 3 -name HHAR -version "shot_portion50" --store "${name}_HHAR_portion50" &
# python main.py -g 3 -name HHAR -version "shot_portion60" --store "${name}_HHAR_portion60"

# wait

# python transfer.py -g 3 -ft True -version "shot_portion35" -shot 10 -name HHAR --pretrained "${name}_HHAR_portion35" &
# python transfer.py -g 3 -ft True -version "shot_portion50" -shot 10 -name HHAR --pretrained "${name}_HHAR_portion50" &
# python transfer.py -g 3 -ft True -version "shot_portion60" -shot 10 -name HHAR --pretrained "${name}_HHAR_portion60" 
# python transfer.py -g 2 -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib"

# wait

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
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" &
