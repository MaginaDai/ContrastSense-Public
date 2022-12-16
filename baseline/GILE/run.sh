name='GILE_v'
for v in 0 1 2 3 4
do
    python main.py -g 3 -name HASC -version "shot${v}" --store "${name}_${v}" &
    python main.py -g 3 -name HHAR -version "shot${v}" --store "${name}_${v}" &
    python main.py -g 3 -name MotionSense -version "shot${v}" --store "${name}_${v}" &
    python main.py -g 3 -name Shoaib -version "shot${v}" --store "${name}_${v}"

    wait

done
