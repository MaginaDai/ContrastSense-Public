store='CALDA_v'
version="shot"

for dataset in "Myo" "NinaPro"
do
    python main.py -g 2 -version "${version}0" --store "${store}0" -shot 10 -name ${dataset} &
    python main.py -g 2 -version "${version}1" --store "${store}1" -shot 10 -name ${dataset} &
    python main.py -g 2 -version "${version}2" --store "${store}2" -shot 10 -name ${dataset} &
    python main.py -g 2 -version "${version}3" --store "${store}3" -shot 10 -name ${dataset} &
    python main.py -g 2 -version "${version}4" --store "${store}4" -shot 10 -name ${dataset} 

    wait
done
