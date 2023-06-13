store='CALDA_w1_v'
version="shot"
w2=1.0
for dataset in "Myo" "NinaPro"
do
    python main.py -w2 ${w2} -g 2 -version "${version}0" --store "${store}0" -shot 10 -name ${dataset} &
    python main.py -w2 ${w2} -g 2 -version "${version}1" --store "${store}1" -shot 10 -name ${dataset} &
    python main.py -w2 ${w2} -g 2 -version "${version}2" --store "${store}2" -shot 10 -name ${dataset} &
    python main.py -w2 ${w2} -g 2 -version "${version}3" --store "${store}3" -shot 10 -name ${dataset} &
    python main.py -w2 ${w2} -g 2 -version "${version}4" --store "${store}4" -shot 10 -name ${dataset} 

    wait
done
