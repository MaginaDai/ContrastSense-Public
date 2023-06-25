version="shot"
slr=0.3
lr=1e-3

for max in 0.001
do
    for lam in 5e3 1e4 5e4
    do
        store="emg_model_v2_cdl_sota"
        # for dataset in "Myo" "NinaPro"
        # do
        #     python main.py -g 0 -lr ${lr} -slr ${slr} -label_type 1 -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        #     python main.py -g 0 -lr ${lr} -slr ${slr} -label_type 1 -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        #     python main.py -g 1 -lr ${lr} -slr ${slr} -label_type 1 -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        #     python main.py -g 1 -lr ${lr} -slr ${slr} -label_type 1 -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        #     python main.py -g 1 -lr ${lr} -slr ${slr} -label_type 1 -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        #     wait
        # done

        store_ft="emg_model_v2_cdl_sota_aug_ewc${lam}_max${max}_"
        for dataset in "Myo" "NinaPro"
        do
            python main_trans_ewc.py -e 400 -ewc True -fishermax ${max} -ewc_lambda ${lam} -aug True -lr 1e-3 -g 0 -ft True -version "${version}0" -shot 10 -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
            python main_trans_ewc.py -e 400 -ewc True -fishermax ${max} -ewc_lambda ${lam} -aug True -lr 1e-3 -g 0 -ft True -version "${version}1" -shot 10 -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
            python main_trans_ewc.py -e 400 -ewc True -fishermax ${max} -ewc_lambda ${lam} -aug True -lr 1e-3 -g 0 -ft True -version "${version}2" -shot 10 -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
            python main_trans_ewc.py -e 400 -ewc True -fishermax ${max} -ewc_lambda ${lam} -aug True -lr 1e-3 -g 1 -ft True -version "${version}3" -shot 10 -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
            python main_trans_ewc.py -e 400 -ewc True -fishermax ${max} -ewc_lambda ${lam} -aug True -lr 1e-3 -g 1 -ft True -version "${version}4" -shot 10 -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4"

            wait
        done
    done
done