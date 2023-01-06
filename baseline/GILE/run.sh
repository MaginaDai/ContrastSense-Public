name='GILE_f1_v'

for v in 0 1 2 3 4
do
    # name="GILE_a${p}_shot${s}_v_0"

    python main.py -g 3 -name HASC -version "shot${v}" -shot 10 --store "${name}_${v}" &
    python main.py -g 3 -name HHAR -version "shot${v}" -shot 10 --store "${name}_${v}" &
    python main.py -g 3 -name MotionSense -version "shot${v}" -shot 10 --store "${name}_${v}" &
    python main.py -g 3 -name Shoaib -version "shot${v}" -shot 10 --store "${name}_${v}"

    wait

done

