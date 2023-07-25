version="shot"
slr=0.3
lr=1e-3
max=0.001
lam=4e3
t=40

store="emg_best_pt_"
# for dataset in "Myo" "NinaPro"
# do
#     python main.py -g 0 -hard True -last_ratio ${r} -time_window ${t} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
#     python main.py -g 0 -hard True -last_ratio ${r} -time_window ${t} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
#     python main.py -g 1 -hard True -last_ratio ${r} -time_window ${t} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
#     python main.py -g 1 -hard True -last_ratio ${r} -time_window ${t} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
#     python main.py -g 1 -hard True -last_ratio ${r} -time_window ${t} -lr ${lr} -slr ${slr} -label_type 0 -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

#     wait
# done

shot=50
for portion in 60 80 100
do
    version="cu_tune_portion_${portion}_shot"
    store="emg_best_pt_"
    store_ft="emg_best_pt_ewc${lam}_portion)_${portion}"
    for dataset in "Myo" "NinaPro"
    do
        python main_trans_ewc.py -fishermax ${max} -ewc_lambda ${lam} -lr 1e-3 -g 0 -ft True -ewc True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python main_trans_ewc.py -fishermax ${max} -ewc_lambda ${lam} -lr 1e-3 -g 0 -ft True -ewc True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python main_trans_ewc.py -fishermax ${max} -ewc_lambda ${lam} -lr 1e-3 -g 0 -ft True -ewc True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python main_trans_ewc.py -fishermax ${max} -ewc_lambda ${lam} -lr 1e-3 -g 1 -ft True -ewc True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python main_trans_ewc.py -fishermax ${max} -ewc_lambda ${lam} -lr 1e-3 -g 1 -ft True -ewc True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4"

        wait
    done
done
