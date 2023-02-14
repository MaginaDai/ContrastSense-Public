# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_plain" --setting 'full' --store "train60_HHAR_supervised_plain" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'full' --store "train60_HHAR_supervised_cross_full" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'sparse' --store "train60_HHAR_supervised_cross_sparse"

# wait

# python main_supervised.py -g 2 -name "HHAR" -version "train70_supervised_plain" --setting 'full' --store "train70_HHAR_supervised_plain" &
# python main_supervised.py -g 2 -name "HHAR" -version "train70_supervised_cross" --setting 'full' --store "train70_HHAR_supervised_cross_full" &
# python main_supervised.py -g 2 -name "HHAR" -version "train45_supervised_plain" --setting 'full' --store "train45_HHAR_supervised_plain" &
# python main_supervised.py -g 2 -name "HHAR" -version "train45_supervised_cross" --setting 'full' --store "train45_HHAR_supervised_cross_full"

# for e in 50 100 200
# do
#     python main_supervised.py -g 2 -name "HHAR" -version "train25_supervised_cross" --setting 'full' --store "train25_HHAR_supervised_cross_full" &
#     python main_supervised.py -g 2 -name "HHAR" -e -version "train25_supervised_cross" --setting 'sparse' --store "train25_HHAR_supervised_cross_sparse"
#     wait
# done

# e=50
# python main_supervised.py -g 2 -e 50 -name "HHAR" -version "train25_supervised_cross" --setting 'sparse' --store "train25_HHAR_supervised_cross_sparse_e50" &
# python main_supervised.py -g 2 -e 100 -name "HHAR" -version "train25_supervised_cross" --setting 'sparse' --store "train25_HHAR_supervised_cross_sparse_e100" &
# python main_supervised.py -g 2 -e 200 -name "HHAR" -version "train25_supervised_cross" --setting 'sparse' --store "train25_HHAR_supervised_cross_sparse_e200"

# wait

# python main_supervised.py -g 2 -e 50 -name "HHAR" -version "train25_supervised_cross" --setting 'full' --store "train25_HHAR_supervised_cross_full_e50" &
# python main_supervised.py -g 2 -e 100 -name "HHAR" -version "train25_supervised_cross" --setting 'full' --store "train25_HHAR_supervised_cross_full_e100" &
# python main_supervised.py -g 2 -e 200 -name "HHAR" -version "train25_supervised_cross" --setting 'full' --store "train25_HHAR_supervised_cross_full_e200"


version="cp"
shot=10
for dataset in "Shoaib"
do
    name="CDL_cp"
    ### with all
    store="CDL_cp_ewc5_"
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -ewc_lambda 5 -version "${version}0" -name ${dataset} --pretrained "${name}0/${dataset}" --store "${store}0" -cross "positions" &
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -ewc_lambda 5 -version "${version}1" -name ${dataset} --pretrained "${name}1/${dataset}" --store "${store}1" -cross "positions" &
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -ewc_lambda 5 -version "${version}2" -name ${dataset} --pretrained "${name}2/${dataset}" --store "${store}2" -cross "positions" &
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -ewc_lambda 5 -version "${version}3" -name ${dataset} --pretrained "${name}3/${dataset}" --store "${store}3" -cross "positions" & 
    python main_trans_ewc.py -shot ${shot} -g 2 -aug True -ewc True -ewc_lambda 5 -version "${version}4" -name ${dataset} --pretrained "${name}4/${dataset}" --store "${store}4" -cross "positions" 
done
