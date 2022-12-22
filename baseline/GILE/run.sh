# name='GILE_shot50_a_v'
v=0

for p in "_l1uo"
do
    for s in 0 10 50
    do
        name="GILE_a${p}_shot${s}_v_0"

        python main.py -g 3 -name HASC -version "a${p}_shot${v}" -shot ${s} --store "${name}" &
        python main.py -g 3 -name HHAR -version "a${p}_shot${v}" -shot ${s} --store "${name}" &
        python main.py -g 3 -name MotionSense -version "a${p}_shot${v}" -shot ${s} --store "${name}" &
        python main.py -g 3 -name Shoaib -version "a${p}_shot${v}" -shot ${s} --store "${name}"

        wait

    done
done

