# name='GILE_s'
# g=2
# for v in 0 1 2 3 4
# do
#     name="GILE_s"

#     python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name HASC &
#     python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name HHAR &
#     python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name MotionSense &
#     python main.py -g ${g} -version "s${v}" -shot 60 --store "${name}${v}" -name Shoaib 

#     wait

# done

# g=0
# name="GILE_cp"
# version="cp"

# python main.py -g ${g} -version "${version}0" -shot 10 -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -shot 10 -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -shot 10 -name Shoaib --store "${name}4"


# g=3
# name="GILE_cu"
# version="shot"

# python main.py -g ${g} -version "${version}0" -shot 10 -name HASC -cross "users" --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -shot 10 -name HASC -cross "users" --store "${name}1"
# wait

# python main.py -g ${g} -version "${version}2" -shot 10 -name HASC -cross "users" --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -shot 10 -name HASC -cross "users" --store "${name}3"
# wait

# python main.py -g ${g} -version "${version}4" -shot 10 -name HASC -cross "users" --store "${name}4"

# name='GILE_cu'
# g=2
# shot=50
# for portion in 60 80 100
# do
#     name="GILE_cu_tune_portion_${portion}_shot"
#     version="tune_portion_${portion}_shot"
#     for v in 0 1 2 3 4
#     do
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HASC &
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HHAR
        
#         wait

#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name MotionSense &
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name Shoaib 

#         wait

#     done
# done

name="GILE_cu"
version="shot"
g=2
shot=100
for v in 0 1 2 3 4
do
    python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HASC &
    python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HHAR &
    python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name MotionSense &
    python main.py -g 2 -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name Shoaib 
    wait

done


##################
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