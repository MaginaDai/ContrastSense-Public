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
# for portion in 60 80 100
# do
#     name="GILE_cd_tune_portion_${portion}_shot"
#     version="cd_tune_portion_${portion}_shot"
#     for shot in 50
#     do
#     python main.py -g ${g} -version "${version}0" -shot ${shot} -name HASC --store "${name}0" -cross "devices" &
#     python main.py -g ${g} -version "${version}1" -shot ${shot} -name HASC --store "${name}1" -cross "devices" &
#     python main.py -g ${g} -version "${version}2" -shot ${shot} -name HASC --store "${name}2" -cross "devices" &
#     python main.py -g ${g} -version "${version}3" -shot ${shot} -name HASC --store "${name}3" -cross "devices" &
#     python main.py -g ${g} -version "${version}4" -shot ${shot} -name HASC --store "${name}4" -cross "devices" 
#     wait

#     done
# done

# g=2
# for portion in 100
# do
#     name="GILE_cp_tune_portion_${portion}_shot"
#     version="cp_tune_portion_${portion}_shot"
#     for shot in 100
#     do
    # python main.py -g ${g} -version "${version}0" -shot ${shot} -name Shoaib --store "${name}0" -cross "positions" &
    # python main.py -g ${g} -version "${version}1" -shot ${shot} -name Shoaib --store "${name}1" -cross "positions" &
    # python main.py -g ${g} -version "${version}2" -shot ${shot} -name Shoaib --store "${name}2" -cross "positions" &
    # wait
    
    # python main.py -g ${g} -version "${version}3" -shot ${shot} -name Shoaib --store "${name}3" -cross "positions" &
    # python main.py -g ${g} -version "${version}4" -shot ${shot} -name Shoaib --store "${name}4" -cross "positions" 
    # wait

#     done
# done

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

# name="GILE_cu"
# version="shot"
# g=3
# shot=1
# for v in 0 1 2 3 4
# do
#     python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HASC &
#     python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HHAR &
#     python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name MotionSense &
#     python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name Shoaib 
#     wait

# done

# python main.py -g ${g} -version "${version}1" -shot ${shot} -cross "users" --store "${name}1" -name HHAR & 
# python main.py -g ${g} -version "${version}0" -shot ${shot} -cross "users" --store "${name}0" -name Shoaib &
# python main.py -g ${g} -version "${version}2" -shot ${shot} -cross "users" --store "${name}2" -name Shoaib 

##################
# version="train25_supervised_cross"
# store="preliminary_25_across"
# g=1
# python main.py -g ${g} -version "${version}" --setting 'full' -shot 0 -name HHAR -cross "users" --store "${store}" &
# python main.py -g ${g} -version "${version}" -shot 5 -name HHAR -cross "users" --store "${store}" &
# python main.py -g ${g} -version "${version}" -shot 10 -name HHAR -cross "users" --store "${store}" &
# python main.py -g ${g} -version "${version}" -shot 50 -name HHAR -cross "users" --store "${store}" &
# python main.py -g ${g} -version "${version}" -shot 100 -name HHAR -cross "users" --store "${store}"


# for a in 45 65
# do
#     name="GILE_alpha${a}"
#     version="alpha${a}_shot"
#     g=3
#     shot=10
#     for v in 0 1 2 3 4
#     do
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HASC &
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name HHAR &
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name MotionSense &
#         python main.py -g ${g} -version "${version}${v}" -shot ${shot} -cross "users" --store "${name}${v}" -name Shoaib 
#         wait

#     done
# done



# name="GILE_users_positions_"
# version="users_positions_shot"
# g=0

# for shot in 100
# do
#     python main.py -g ${g} -version "${version}0" -shot ${shot} -cross "multiple" --store "${name}0" -name Shoaib &
#     python main.py -g ${g} -version "${version}1" -shot ${shot} -cross "multiple" --store "${name}1" -name Shoaib &
#     python main.py -g ${g} -version "${version}2" -shot ${shot} -cross "multiple" --store "${name}2" -name Shoaib &
#     python main.py -g ${g} -version "${version}3" -shot ${shot} -cross "multiple" --store "${name}3" -name Shoaib &
#     python main.py -g ${g} -version "${version}4" -shot ${shot} -cross "multiple" --store "${name}4" -name Shoaib 
#     wait

# done


# name="GILE_users_devices_"
# version="users_devices_shot"
# g=0

# for shot in 100
# do
#     python main.py -g ${g} -version "${version}0" -shot ${shot} -cross "multiple" --store "${name}0" -name HASC &
#     python main.py -g ${g} -version "${version}1" -shot ${shot} -cross "multiple" --store "${name}1" -name HASC &
#     python main.py -g ${g} -version "${version}2" -shot ${shot} -cross "multiple" --store "${name}2" -name HASC &
#     python main.py -g ${g} -version "${version}3" -shot ${shot} -cross "multiple" --store "${name}3" -name HASC &
#     python main.py -g ${g} -version "${version}4" -shot ${shot} -cross "multiple" --store "${name}4" -name HASC 
#     wait

# done


name="GILE_leave_shot"
version="leave_shot"
g=0
shot=10
for dataset in "HASC" "MotionSense"
do
    python main.py -g ${g} -version "${version}0" -shot ${shot} -cross "users" --store "${name}0" -name ${dataset} &
    python main.py -g ${g} -version "${version}1" -shot ${shot} -cross "users" --store "${name}1" -name ${dataset} &
    python main.py -g ${g} -version "${version}2" -shot ${shot} -cross "users" --store "${name}2" -name ${dataset} &
    python main.py -g ${g} -version "${version}3" -shot ${shot} -cross "users" --store "${name}3" -name ${dataset} &
    python main.py -g ${g} -version "${version}4" -shot ${shot} -cross "users" --store "${name}4" -name ${dataset} 
    wait
    
done