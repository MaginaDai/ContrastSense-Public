name='Ori'

for lr in 0.0001 0.00001 0.000001
do
    python main.py -g 0 -lr ${lr} -name HASC --store "${name}_lr${lr}_HASC" &
    python main.py -g 0 -lr ${lr} -name HHAR --store "${name}_lr${lr}_HHAR" &
    python main.py -g 1 -lr ${lr} -name MotionSense --store "${name}_lr${lr}_MotionSense" &
    python main.py -g 1 -lr ${lr} -name Shoaib --store "${name}_lr${lr}_Shoaib"
    wait
done

