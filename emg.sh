version="shot"
store="no"
store_ft="emg_model_v2_plain_"

for dataset in "Myo" "NinaPro"
do
    python main_trans_ewc.py -g 0 -ft True -version "${version}0" -shot 10 -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
    python main_trans_ewc.py -g 0 -ft True -version "${version}1" -shot 10 -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
    python main_trans_ewc.py -g 0 -ft True -version "${version}2" -shot 10 -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
    python main_trans_ewc.py -g 1 -ft True -version "${version}3" -shot 10 -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
    python main_trans_ewc.py -g 1 -ft True -version "${version}4" -shot 10 -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4"

    wait
done