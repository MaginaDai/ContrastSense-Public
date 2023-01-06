# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_plain" --setting 'full' --store "train60_HHAR_supervised_plain" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'full' --store "train60_HHAR_supervised_cross_full" &
# python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'sparse' --store "train60_HHAR_supervised_cross_sparse"

# wait

python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'full' --store "train60_HHAR_supervised_cross_full" &
python main_supervised.py -g 2 -name "HHAR" -version "train60_supervised_cross" --setting 'sparse' --store "train60_HHAR_supervised_cross_sparse"

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
