version="shot"

slr=0.7
shot=10

for r in 0.05
do
    store="hard_v3_cdl_r${r}_"
    for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
    do
        python main.py -g 0 -hard True -sample_ratio ${r} -label_type 1 -slr ${slr} -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        python main.py -g 0 -hard True -sample_ratio ${r} -label_type 1 -slr ${slr} -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        python main.py -g 0 -hard True -sample_ratio ${r} -label_type 1 -slr ${slr} -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        python main.py -g 1 -hard True -sample_ratio ${r} -label_type 1 -slr ${slr} -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        python main.py -g 1 -hard True -sample_ratio ${r} -label_type 1 -slr ${slr} -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        wait
    done



    for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
    do
        python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}0" -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}0" &
        python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}1" -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}1" &
        python main_trans_ewc.py -shot ${shot} -g 0 -version "${version}2" -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}2" &
        python main_trans_ewc.py -shot ${shot} -g 1 -version "${version}3" -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}3" &
        python main_trans_ewc.py -shot ${shot} -g 1 -version "${version}4" -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}4"
        
        wait
    done

done