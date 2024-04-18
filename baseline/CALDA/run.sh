# store='CALDA_w1_v'
# version="shot"
# w2=1.0
# for dataset in "Myo" "NinaPro"
# do
#     python main.py -w2 ${w2} -g 2 -version "${version}0" --store "${store}0" -shot 10 -name ${dataset} &
#     python main.py -w2 ${w2} -g 2 -version "${version}1" --store "${store}1" -shot 10 -name ${dataset} &
#     python main.py -w2 ${w2} -g 2 -version "${version}2" --store "${store}2" -shot 10 -name ${dataset} &
#     python main.py -w2 ${w2} -g 2 -version "${version}3" --store "${store}3" -shot 10 -name ${dataset} &
#     python main.py -w2 ${w2} -g 2 -version "${version}4" --store "${store}4" -shot 10 -name ${dataset} 

#     wait
# done


# store='CALDA_w1_v'
# version="shot"
# w2=1.0
# for shot in 1 5 10 50
# do
#     for dataset in "Myo" "NinaPro"
#     do
#         python main.py -w2 ${w2} -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 1 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 1 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset} 

#         wait
#     done
# done

# shot=50
# for portion in 60 80 100
# do
#     version="cu_tune_portion_${portion}_shot"
#     store="CALDA_w1_portion_${portion}_"
    
#     for dataset in "Myo" "NinaPro"
#     do
#         python main.py -w2 ${w2} -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 1 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
#         python main.py -w2 ${w2} -g 1 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset} 

#         wait
#     done

# done


w2=1.0
shot=10
for a in 45 65
do
    store="CALDA_alpha${a}_"
    store_ft="CALDA_alpha${a}_"
    version="alpha${a}_shot"

    for dataset in "Myo" "NinaPro"
    do
        python main.py -w2 ${w2} -g 0 -version "${version}0" --store "${store}0" -shot ${shot} -name ${dataset} &
        python main.py -w2 ${w2} -g 0 -version "${version}1" --store "${store}1" -shot ${shot} -name ${dataset} &
        python main.py -w2 ${w2} -g 0 -version "${version}2" --store "${store}2" -shot ${shot} -name ${dataset} &
        python main.py -w2 ${w2} -g 1 -version "${version}3" --store "${store}3" -shot ${shot} -name ${dataset} &
        python main.py -w2 ${w2} -g 1 -version "${version}4" --store "${store}4" -shot ${shot} -name ${dataset} 

        wait
    done

done
