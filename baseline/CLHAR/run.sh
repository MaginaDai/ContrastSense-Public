# name='CLHAR_s'
# version="cp"
# g=0

# for v in 0 1 2 3 4
# do
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name HASC &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name HHAR &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name MotionSense &
#     python main.py -g ${g} --store "${name}${v}" -version "${version}${v}" -name Shoaib 

#     wait

#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name HASC --pretrained "${name}${v}/HASC" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name HHAR --pretrained "${name}${v}/HHAR" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name MotionSense --pretrained "${name}${v}/MotionSense" --store "${name}${v}" &
#     python transfer.py -g ${g} -ft True -version "${version}${v}" -shot 10 -name Shoaib --pretrained "${name}${v}/Shoaib" --store "${name}${v}" 

#     wait

# done


# name='CLHAR_cp'
# version="cp"
# g=1

# python main.py -g ${g} -version "${version}0" -name Shoaib --store "${name}0" &
# python main.py -g ${g} -version "${version}1" -name Shoaib --store "${name}1" &
# python main.py -g ${g} -version "${version}2" -name Shoaib --store "${name}2" &
# python main.py -g ${g} -version "${version}3" -name Shoaib --store "${name}3" &
# python main.py -g ${g} -version "${version}4" -name Shoaib --store "${name}4" &

# wait

# python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name Shoaib --pretrained "${name}0/Shoaib" --store "${name}0" &
# python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name Shoaib --pretrained "${name}1/Shoaib" --store "${name}1" &
# python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name Shoaib --pretrained "${name}2/Shoaib" --store "${name}2" &
# python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name Shoaib --pretrained "${name}3/Shoaib" --store "${name}3" &
# python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name Shoaib --pretrained "${name}4/Shoaib" --store "${name}4" 

# wait

name='CLHAR_cd'
version="cd"
g=1

python main.py -g ${g} -version "${version}0" -name HHAR --store "${name}0" &
python main.py -g ${g} -version "${version}1" -name HHAR --store "${name}1" &
python main.py -g ${g} -version "${version}2" -name HHAR --store "${name}2" &
python main.py -g ${g} -version "${version}3" -name HHAR --store "${name}3" &
python main.py -g ${g} -version "${version}4" -name HHAR --store "${name}4" &

wait

python transfer.py -g ${g} -ft True -version "${version}0" -shot 10 -name HHAR --pretrained "${name}0/HHAR" --store "${name}0" &
python transfer.py -g ${g} -ft True -version "${version}1" -shot 10 -name HHAR --pretrained "${name}1/HHAR" --store "${name}1" &
python transfer.py -g ${g} -ft True -version "${version}2" -shot 10 -name HHAR --pretrained "${name}2/HHAR" --store "${name}2" &
python transfer.py -g ${g} -ft True -version "${version}3" -shot 10 -name HHAR --pretrained "${name}3/HHAR" --store "${name}3" &
python transfer.py -g ${g} -ft True -version "${version}4" -shot 10 -name HHAR --pretrained "${name}4/HHAR" --store "${name}4" 

wait
