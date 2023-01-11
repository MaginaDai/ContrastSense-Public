g=3
v=3
for v in 3
do
    for b in 32 64
    do
        store="Mixup_v${v}_b${b}"
        python main.py -b ${b} -g 0 -version "shot${v}" --store ${store} -name 'HASC' &
        python main.py -b ${b} -g 0 -version "shot${v}" --store ${store} -name 'HHAR' &
        python main.py -b ${b} -g 1 -version "shot${v}" --store ${store} -name 'MotionSense' &
        python main.py -b ${b} -g 1 -version "shot${v}" --store ${store} -name 'Shoaib'

        wait
    done
done

