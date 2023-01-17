name='GILE_s'
g=2
for v in 0 1 2 3 4
do
    name="GILE_s"

    python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name HASC &
    python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name HHAR &
    python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name MotionSense &
    python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name Shoaib 

    wait

done

# 25 45 65
# 10 50 100 200 500

# v=0
# store="GILE_full_version_v${v}"

# python main.py -g 1 -name HHAR -version "train25_alltune_cross_v${v}" --setting full --store "${store}_25" &
# python main.py -g 1 -name HHAR -version "train65_alltune_cross_v${v}" --setting full --store "${store}_65"

# wait

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