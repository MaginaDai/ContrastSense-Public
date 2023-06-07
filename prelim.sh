# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_plain" --setting 'full' --store "train60_HHAR_supervised_plain" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'full' --store "train60_HHAR_supervised_cross_full" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'sparse' --store "train60_HHAR_supervised_cross_sparse"

# wait

# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train65_supervised_random" --setting 'full' --store "train65_HHAR_supervised_random_lr5-5" &
# python main_supervised.py -g 3 -name "HHAR" -lr 0.00005 -version "train65_supervised_cross" --setting 'full' --store "train65_HHAR_supervised_cross_full_lr5-5" 

# wait

# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train45_supervised_random" --setting 'full' --store "train45_HHAR_supervised_random_lr5-5" &
# python main_supervised.py -g 3 -name "HHAR" -lr 0.00005 -version "train45_supervised_cross" --setting 'full' --store "train45_HHAR_supervised_cross_full_lr5-5" 

# wait
# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_random" --setting 'full' --store "train25_HHAR_supervised_random_lr5-5" &
# python main_supervised.py -g 3 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'full' --store "train25_HHAR_supervised_cross_full_lr5-5"

# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_label" --setting 'sparse' -shot 0 --store "train25_HHAR_supervised_label_lr5-5" &
# python main_supervised.py -g 3 -name "HHAR" -lr 0.00005 -version "train45_supervised_label" --setting 'sparse' -shot 0 --store "train45_HHAR_supervised_label_lr5-5" &
# python main_supervised.py -g 3 -name "HHAR" -lr 0.00005 -version "train65_supervised_label" --setting 'sparse' -shot 0 --store "train65_HHAR_supervised_label_lr5-5"

# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'sparse' -shot 5 --store "train25_supervised_cross" &
python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'sparse' -shot 10 --store "train25_supervised_cross" &
# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'sparse' -shot 50 --store "train25_supervised_cross" &
# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'sparse' -shot 100 --store "train25_supervised_cross" &
# python main_supervised.py -g 2 -name "HHAR" -lr 0.00005 -version "train25_supervised_cross" --setting 'full' -shot 0 --store "train25_supervised_cross" &

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


# python main_supervised.py -g 2 -name "HHAR" -version "domain_shift" --setting 'sparse' -shot 0 --store "CPC_domain_shift" &