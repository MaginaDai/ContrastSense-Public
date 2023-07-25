version="test"

for shot in 0
do
    store="SACL_new_test_"
    store_ft="SACL_new_test_"
    # for dataset in "SEED_IV"
    # do
    #     python main.py -g 0 -version "${version}0" -name ${dataset} --store "${store}0" &
    #     python main.py -g 0 -version "${version}1" -name ${dataset} --store "${store}1" &
    #     python main.py -g 0 -version "${version}2" -name ${dataset} --store "${store}2" &
    #     python main.py -g 1 -version "${version}3" -name ${dataset} --store "${store}3" &
    #     python main.py -g 1 -version "${version}4" -name ${dataset} --store "${store}4"

    #     wait
    # done

    for dataset in "SEED_IV"
    do
        python transfer.py -g 0 -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
        python transfer.py -g 0 -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
        python transfer.py -g 0 -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
        python transfer.py -g 1 -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
        python transfer.py -g 1 -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

        wait
    done
done
