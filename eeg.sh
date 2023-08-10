version="shot"
slr=0.9
lr=5e-5
max=0.001
lam=4e3
window=0
shot=10
b=512

for r in 0.2 0.4
do
    store="eeg/cl_r_${r}"
    for dataset in "sleepEDF"
    do
        python main.py -g 0 -b ${b} -last_ratio ${r} -time_window ${window} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        python main.py -g 0 -b ${b} -last_ratio ${r} -time_window ${window} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        python main.py -g 1 -b ${b} -last_ratio ${r} -time_window ${window} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        python main.py -g 1 -b ${b} -last_ratio ${r} -time_window ${window} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        python main.py -g 1 -b ${b} -last_ratio ${r} -time_window ${window} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        wait
    done

    store_ft="eeg/cl_r_${r}"
    for dataset in "sleepEDF"
    do
        python main_trans_ewc.py -lr 1e-4 -fishermax ${max} -ewc_lambda ${lam} -g 0 -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python main_trans_ewc.py -lr 1e-4 -fishermax ${max} -ewc_lambda ${lam} -g 0 -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python main_trans_ewc.py -lr 1e-4 -fishermax ${max} -ewc_lambda ${lam} -g 0 -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python main_trans_ewc.py -lr 1e-4 -fishermax ${max} -ewc_lambda ${lam} -g 1 -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python main_trans_ewc.py -lr 1e-4 -fishermax ${max} -ewc_lambda ${lam} -g 1 -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4"

        wait 
    done
done

