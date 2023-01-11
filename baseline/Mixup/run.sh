g=3
v=3
for v in 3
do
    for lr in 0.01 0.05 0.1 
    do
        store="Mixup_v${v}_lr${lr}"
        python main.py -lr ${lr} -g 0 -version "shot${v}" --store ${store} -name 'HASC' &
        python main.py -lr ${lr} -g 0 -version "shot${v}" --store ${store} -name 'HHAR' &
        python main.py -lr ${lr} -g 1 -version "shot${v}" --store ${store} -name 'MotionSense' &
        python main.py -lr ${lr} -g 1 -version "shot${v}" --store ${store} -name 'Shoaib'

        wait
    done
done

