# g=2
# for v in 0 1 2 3 4
# do

#     store="Mixup_s${v}"
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'HASC' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'HHAR' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'MotionSense' &
#     python main.py -g 2 -version "s${v}" --store ${store} -name 'Shoaib'

#     wait

# done

# g=3
# name="mixup_cp_adpt"
# version="cp"

# python main.py -g ${g} -version "${version}0" -shot 10 -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name Shoaib --store "${name}4"

# wait

# g=3
# name="mixup_cd_adpt"
# version="cd"

# python main.py -g ${g} -version "${version}0" -shot 10 -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name HASC --store "${name}4"

# wait

# g=3
# name="mixup_cu_adpt"
# version="shot"

# python main.py -g ${g} -version "${version}0" -shot 10 -name HASC --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name HASC --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name HASC --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name HASC --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name HASC --store "${name}4"


# version="shot"
# shot=50
# for portion in 60 80 100
# do
#     version="tune_portion_${portion}_shot"
#     for v in 0 1 2 3 4
#     do
#         store="mixup_cu_adpt_tune_portion_${portion}_${v}"
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HASC' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HHAR' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'MotionSense' &
#         python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'Shoaib'
#         wait

#     done
# done

version="shot"
for shot in 100
do
    for v in 0 1 2 3 4
    do
        store="mixup_cu${v}"
        python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HASC' &
        python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'HHAR' &
        python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'MotionSense' &
        python main.py -g 3 -version "${version}${v}" --store ${store} -shot ${shot} -name 'Shoaib'
        wait
    done
done