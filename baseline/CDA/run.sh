
store="ConSSL_lr1e-3_co_v"
store_ft="ConSSL_lr1e-3_co_v"
version="shot"
lr=1e-3
for dataset in "Myo" "NinaPro"
do
    python main.py -lr ${lr} -g 0 -version "${version}0" --store "${store}0" -name ${dataset} &
    python main.py -lr ${lr} -g 0 -version "${version}1" --store "${store}1" -name ${dataset} &
    python main.py -lr ${lr} -g 0 -version "${version}2" --store "${store}2" -name ${dataset} &
    python main.py -lr ${lr} -g 1 -version "${version}3" --store "${store}3" -name ${dataset} &
    python main.py -lr ${lr} -g 1 -version "${version}4" --store "${store}4" -name ${dataset} 

    wait
done

version="shot"


for shot in 1 5 10 50
do
    store="ConSSL_lr1e-3_co_v"
    store_ft="ConSSL_lr1e-3_co_v"
    for dataset in "Myo" "NinaPro"
    do
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python transfer.py -g 1 -ft True -lr 1e-2 -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python transfer.py -g 1 -ft True -lr 1e-2 -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

        wait
    done
done

shot=50
for portion in 60 80 100
do
    version="cu_tune_portion_${portion}_shot"
    store="ConSSL_lr1e-3_co_v"
    store_ft="ConSSL_lr1e-3_co_portion_${portion}_v"
    for dataset in "Myo" "NinaPro"
    do
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python transfer.py -g 0 -ft True -lr 1e-2 -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python transfer.py -g 1 -ft True -lr 1e-2 -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python transfer.py -g 1 -ft True -lr 1e-2 -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

        wait
    done
done