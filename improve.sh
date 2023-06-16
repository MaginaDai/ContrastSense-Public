version="shot"
store="hard_v1"
slr=0.7
shot=10

for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    python main.py -g 2 -hard True -label_type 1 -slr ${slr} -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
    python main.py -g 2 -hard True -label_type 1 -slr ${slr} -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
    python main.py -g 2 -hard True -label_type 1 -slr ${slr} -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
    python main.py -g 3 -hard True -label_type 1 -slr ${slr} -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
    python main.py -g 3 -hard True -label_type 1 -slr ${slr} -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

    wait
done



for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
do
    python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}0" -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}_${ewc}_0" &
    python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}1" -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}_${ewc}_1" &
    python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}2" -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}_${ewc}_2" &
    python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}_${ewc}_3" &
    python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}_${ewc}_4"
    
    wait
done