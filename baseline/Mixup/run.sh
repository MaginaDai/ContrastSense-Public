g=2
for v in 0 1 2 3 4
do

    store="Mixup_s${v}"
    python main.py -g 2 -version "s${v}" --store ${store} -name 'HASC' &
    python main.py -g 2 -version "s${v}" --store ${store} -name 'HHAR' &
    python main.py -g 2 -version "s${v}" --store ${store} -name 'MotionSense' &
    python main.py -g 2 -version "s${v}" --store ${store} -name 'Shoaib'

    wait

done

