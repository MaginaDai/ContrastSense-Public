# for lr in 1e-6 1e-5 1e-4
# do
#     store="CDA_lr${lr}_v"
#     version="shot"
#     w2=1.0
#     for dataset in "Myo" "NinaPro"
#     do
#         python main.py -lr ${lr} -g 2 -version "${version}0" --store "${store}0" -name ${dataset} &
#         python main.py -lr ${lr} -g 2 -version "${version}1" --store "${store}1" -name ${dataset} &
#         python main.py -lr ${lr} -g 2 -version "${version}2" --store "${store}2" -name ${dataset} &
#         python main.py -lr ${lr} -g 3 -version "${version}3" --store "${store}3" -name ${dataset} &
#         python main.py -lr ${lr} -g 3 -version "${version}4" --store "${store}4" -name ${dataset} 

#         wait
#     done
# done

g=3
for lr in 1e-6 1e-5 1e-4
do
    name="CDA_lr${lr}_v"
    store="CDA_lr${lr}_v"
    version="shot"
    w2=1.0
    for dataset in "Myo" "NinaPro"
    do
        python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name ${dataset} --pretrained "${name}0/${dataset}" --store "${name}0" &
        python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name ${dataset} --pretrained "${name}1/${dataset}" --store "${name}1" &
        python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name ${dataset} --pretrained "${name}2/${dataset}" --store "${name}2" &
        python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name ${dataset} --pretrained "${name}3/${dataset}" --store "${name}3" &
        python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name ${dataset} --pretrained "${name}4/${dataset}" --store "${name}4" 

        wait

    done
done