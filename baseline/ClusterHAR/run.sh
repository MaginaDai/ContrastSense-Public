#!/bin/bash

# python main.py -g 0 -do_cluster True -name HASC --store "${name}_HASC" &
# python main.py -g 0 -do_cluster True -name HHAR --store "${name}_HHAR" &
# python main.py -g 1 -do_cluster True -name MotionSense --store "${name}_MotionSense" &
# python main.py -g 1 -do_cluster True -name Shoaib --store "${name}_Shoaib"


# python transfer.py -g 0 -shot 10 -version 'shot' -name HASC --pretrained "HASC" &
# python transfer.py -g 0 -shot 10 -version 'shot' -name HHAR --pretrained "HHAR" &
# python transfer.py -g 1 -shot 10 -version 'shot' -name MotionSense --pretrained "MotionSense" &
# python transfer.py -g 1 -shot 10 -version 'shot' -name Shoaib --pretrained "Shoaib" 

# python transfer.py -g 0 -lr 0.01 -shot 10 -version 'shot' -name HASC --pretrained "${name}_HASC" &
# python transfer.py -g 0 -lr 0.01 -shot 10 -version 'shot' -name HHAR --pretrained "${name}_HHAR" &
# python transfer.py -g 1 -lr 0.01 -shot 10 -version 'shot' -name MotionSense --pretrained "${name}_MotionSense" &
# python transfer.py -g 1 -lr 0.01 -shot 10 -version 'shot' -name Shoaib --pretrained "${name}_Shoaib"


# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HASC --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name HHAR --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_MotionSense" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name MotionSense --pretrained "${name}_Shoaib" 

# wait

# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HASC" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_HHAR" &
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_MotionSense"
# python transfer.py -g 2 -ft True -version shot -shot 10 -name Shoaib --pretrained "${name}_Shoaib" &



# python main.py -g 0 -do_cluster True -name HASC --store "${name}_HASC" &
# python main.py -g 0 -do_cluster True -name HHAR --store "${name}_HHAR" &
# python main.py -g 1 -do_cluster True -name MotionSense --store "${name}_MotionSense" &
# python main.py -g 1 -do_cluster True -name Shoaib --store "${name}_Shoaib"


# g=2
# version="shot"
# store="ClusterCL_"
# store_ft="ClusterCL_"
# shot=10
# for shot in 1 5 50
# do
#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         # python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

#         # wait

#         python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#         python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#         python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#         python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#         python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#         wait

#     done
# done

# shot=50
# for portion in 60 80 100
# do
#     version="tune_portion_${portion}_shot"
#     store_ft="ClusterCL_tune_portion_${portion}_shot"
#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         # python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
#         # python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

#         # wait

#         python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#         python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#         python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#         python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#         python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#         wait

#     done
# done


# g=2
# version="cp"
# store="ClusterCL_cp_"
# store_ft="ClusterCL_cp_"
# shot=10
# dataset="Shoaib"
# for dataset in "Shoaib"
# do
#     python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
#     python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
#     python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
#     python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
#     python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

#     wait
# done

# for shot in 5 10 50 100
# do
#     python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#     python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#     python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#     python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#     python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#     wait

# done

# shot=50
# for portion in 100
# do
#     version="cp_tune_portion_100_shot"
#     store_ft="ClusterCL_cp_tune_portion_100_"

#     python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#     python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#     python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#     python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#     python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#     wait

# done


# g=2
# version="cd"
# store="ClusterCL_cd_"
# store_ft="ClusterCL_cd_"
# shot=10

# for dataset in "HASC"
# do
#     python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
#     python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
#     python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
#     python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
#     python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

#     wait
# done

# dataset="HASC"
# for shot in 5 10 50 100
# do
#     python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#     python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#     python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#     python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#     python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#     wait

# done

# shot=50
# for portion in 60 80 100
# do
#     version="cd_tune_portion_${portion}_shot"
#     store_ft="ClusterCL_cd_tune_portion_${portion}_"

#     python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#     python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#     python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#     python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#     python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#     wait

# done


# g=3

# for a in 45 65
# do
#     version="alpha${a}_shot"
#     store="ClusterCL_alpha${a}_"
#     store_ft="ClusterCL_alpha${a}_"
#     shot=10

#     for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
#     do
#         python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
#         python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
#         python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
#         python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
#         python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

#         wait

#         python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
#         python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
#         python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
#         python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
#         python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

#         wait

#     done
# done

version="users_devices_shot"
store="ClusterCL_users_devices_"
store_ft="ClusterCL_users_devices_"
g=1
for dataset in "HASC"
do
    python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
    python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
    python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
    python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
    python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

    wait
done

dataset="HASC"
for shot in 1 5 10 20 50
do
    python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
    python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
    python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
    python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
    python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

    wait
done

version="users_positions_shot"
store="ClusterCL_users_positions_"
store_ft="ClusterCL_users_positions_"
g=1
for dataset in "Shoaib"
do
    python main.py -g ${g} --store "${store}0" -version "${version}0" -name ${dataset} &
    python main.py -g ${g} --store "${store}1" -version "${version}1" -name ${dataset} &
    python main.py -g ${g} --store "${store}2" -version "${version}2" -name ${dataset} &
    python main.py -g ${g} --store "${store}3" -version "${version}3" -name ${dataset} &
    python main.py -g ${g} --store "${store}4" -version "${version}4" -name ${dataset} 

    wait
done

dataset="Shoaib"
for shot in 1 5 10 20 50
do
    python transfer.py -g ${g} -ft True -version "${version}0" -shot ${shot} -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store_ft}0" &
    python transfer.py -g ${g} -ft True -version "${version}1" -shot ${shot} -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store_ft}1" &
    python transfer.py -g ${g} -ft True -version "${version}2" -shot ${shot} -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store_ft}2" &
    python transfer.py -g ${g} -ft True -version "${version}3" -shot ${shot} -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store_ft}3" &
    python transfer.py -g ${g} -ft True -version "${version}4" -shot ${shot} -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store_ft}4" 

    wait
done