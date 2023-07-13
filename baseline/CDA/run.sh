for lr in 1e-3
do
    store="ConSSL_lr${lr}_co_v"
    store_ft="ConSSL_lr${lr}_co_ft_1e-3_v"
    version="shot"
    for dataset in "Myo" "NinaPro"
    do
        python main.py -lr ${lr} -g 0 -version "${version}0" --store "${store}0" -name ${dataset} &
        python main.py -lr ${lr} -g 0 -version "${version}1" --store "${store}1" -name ${dataset} &
        python main.py -lr ${lr} -g 0 -version "${version}2" --store "${store}2" -name ${dataset} &
        python main.py -lr ${lr} -g 1 -version "${version}3" --store "${store}3" -name ${dataset} &
        python main.py -lr ${lr} -g 1 -version "${version}4" --store "${store}4" -name ${dataset} 

        wait
    done
    
    for dataset in "Myo" "NinaPro"
    do
        python transfer.py -g 0 -ft True -lr 1e-3 -version "${version}0" -shot 10 -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python transfer.py -g 0 -ft True -lr 1e-3 -version "${version}1" -shot 10 -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python transfer.py -g 0 -ft True -lr 1e-3 -version "${version}2" -shot 10 -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python transfer.py -g 1 -ft True -lr 1e-3 -version "${version}3" -shot 10 -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python transfer.py -g 1 -ft True -lr 1e-3 -version "${version}4" -shot 10 -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

        wait
    done
done