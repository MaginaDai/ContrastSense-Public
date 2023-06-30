version="shot"

slr=0.7
shot=10

# for r in 10
# do
    # store="hard_v6_cl_t${r}_"
    # for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
    # do
    #     python main.py -g 2 -hard True -time_window ${r} -label_type 0 -slr ${slr} -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
    #     python main.py -g 2 -hard True -time_window ${r} -label_type 0 -slr ${slr} -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
    #     python main.py -g 3 -hard True -time_window ${r} -label_type 0 -slr ${slr} -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
    #     python main.py -g 3 -hard True -time_window ${r} -label_type 0 -slr ${slr} -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
    #     python main.py -g 3 -hard True -time_window ${r} -label_type 0 -slr ${slr} -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

    #     wait
    # done



    # for dataset in "HASC"
    # do
        # python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}0" -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}0" &
        # python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}1" -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}1" &
        # python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}2" -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}2" &
        # python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}3" &
        # python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}4"
        
        # wait
    # done

# done

store="hard_v6_cl_t10_"
python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}0" -name HASC --pretrained "${store}0/HASC" --store "${store}0" &
python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}2" -name MotionSense --pretrained "${store}2/MotionSense" --store "${store}2" &
python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}3" -name MotionSense --pretrained "${store}3/MotionSense" --store "${store}3" &
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name HHAR --pretrained "${store}3/HHAR" --store "${store}3" &
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name HASC --pretrained "${store}4/HASC" --store "${store}4" &
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name Shoaib --pretrained "${store}4/Shoaib" --store "${store}4"

store="hard_v6_cl_t30_"
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name HASC --pretrained "${store}3/HASC" --store "${store}3" &
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name HASC --pretrained "${store}4/HASC" --store "${store}4" &
python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name MotionSense --pretrained "${store}4/MotionSense" --store "${store}4" &
