name='GILE_f1_v'

# for v in 0 1 2 3 4
# do
#     # name="GILE_a${p}_shot${s}_v_0"

#     python main.py -g 3 -name HASC -version "shot${v}" -shot 10 --store "${name}_${v}" &
#     python main.py -g 3 -name HHAR -version "shot${v}" -shot 10 --store "${name}_${v}" &
#     python main.py -g 3 -name MotionSense -version "shot${v}" -shot 10 --store "${name}_${v}" &
#     python main.py -g 3 -name Shoaib -version "shot${v}" -shot 10 --store "${name}_${v}"

#     wait

# done

# 25 45 65
# 10 50 100 200 500

v=0
store="GILE_full_version_v${v}"

python main.py -g 1 -name HHAR -version "train25_alltune_cross_v${v}" --setting full --store "${store}_25" &
python main.py -g 1 -name HHAR -version "train65_alltune_cross_v${v}" --setting full --store "${store}_65"

wait

# for v in 2 3 4
# do
#     for shot in 10 50 200 500
#     do 
#         store="GILE_shot${shot}_version_v${v}"

#         python main.py -g 1 -name HHAR -version "train25_alltune_cross_v${v}" -shot ${shot} --store "${store}_25" &
#         python main.py -g 1 -name HHAR -version "train45_alltune_cross_v${v}" -shot ${shot} --store "${store}_45" &
#         python main.py -g 1 -name HHAR -version "train65_alltune_cross_v${v}" -shot ${shot} --store "${store}_65"
        
#         wait
#     done

#     store="GILE_full_version_v${v}"

#     python main.py -g 1 -name HHAR -version "train25_alltune_cross_v${v}" --setting full --store "${store}_25" &
#     python main.py -g 1 -name HHAR -version "train45_alltune_cross_v${v}" --setting full --store "${store}_45" &
#     python main.py -g 1 -name HHAR -version "train65_alltune_cross_v${v}" --setting full --store "${store}_65"
    
#     wait
# done