# name="Shot"
# version="shot"
# # e=400
# ewc=25
# lr=0.00025
# for e in 200 300 600
# do
#     for v in 0 1 2 3 4
#     do
#         store="f1_e${e}_v${v}"
#         python main_trans_ewc.py -g 1 -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}_${v}_ewc_pretrain/HASC" --store ${store} &
#         python main_trans_ewc.py -g 1 -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}_${v}_ewc_pretrain/HHAR" --store ${store} &
#         python main_trans_ewc.py -g 1 -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}_${v}_ewc_pretrain/Shoaib" --store ${store} &
#         python main_trans_ewc.py -g 1 -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}_${v}_ewc_pretrain/MotionSense" --store ${store} 

#         wait
#     done
# done

# version="shot"
# e=400
# ewc=1
# lr=0.0005
# v=2
# for k in 512
# do 
#     name="MoCo_K${k}_v${v}"

#     python main.py -g 3 -moco_K ${k} -label_type 1 -version "shot${v}" -name HASC --store "${name}" &
#     python main.py -g 3 -moco_K ${k} -label_type 1 -version "shot${v}" -name HHAR --store "${name}" &
#     python main.py -g 2 -moco_K ${k} -label_type 1 -version "shot${v}" -name MotionSense --store "${name}" &
#     python main.py -g 2 -moco_K ${k} -label_type 1 -version "shot${v}" -name Shoaib --store "${name}"

#     wait
    
#     store="MoCo_K${k}_v${v}"

#     python main_trans_ewc.py -g 2 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
#     python main_trans_ewc.py -g 2 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
#     python main_trans_ewc.py -g 3 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
#     python main_trans_ewc.py -g 3 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

#     wait
# done

version="shot"

slr=0.7
shot=10

for r in 0.30 0.50
do
    store="hard_v5_crt_cl_r${r}_"
    for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
    do
        python main.py -g 2 -hard True -last_ratio ${r} -label_type 0 -slr ${slr} -version "${version}0" -name ${dataset} --store "${store}0" -cross "users" &
        python main.py -g 2 -hard True -last_ratio ${r} -label_type 0 -slr ${slr} -version "${version}1" -name ${dataset} --store "${store}1" -cross "users" &
        python main.py -g 2 -hard True -last_ratio ${r} -label_type 0 -slr ${slr} -version "${version}2" -name ${dataset} --store "${store}2" -cross "users" &
        python main.py -g 3 -hard True -last_ratio ${r} -label_type 0 -slr ${slr} -version "${version}3" -name ${dataset} --store "${store}3" -cross "users" &
        python main.py -g 3 -hard True -last_ratio ${r} -label_type 0 -slr ${slr} -version "${version}4" -name ${dataset} --store "${store}4" -cross "users" 

        wait
    done



    for dataset in "HASC" "HHAR" "MotionSense" "Shoaib"
    do
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}0" -name ${dataset} --pretrained "${store}0/${dataset}" --store "${store}0" &
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}1" -name ${dataset} --pretrained "${store}1/${dataset}" --store "${store}1" &
        python main_trans_ewc.py -shot ${shot} -g 2 -version "${version}2" -name ${dataset} --pretrained "${store}2/${dataset}" --store "${store}2" &
        python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}3" -name ${dataset} --pretrained "${store}3/${dataset}" --store "${store}3" &
        python main_trans_ewc.py -shot ${shot} -g 3 -version "${version}4" -name ${dataset} --pretrained "${store}4/${dataset}" --store "${store}4"
        
        wait
    done

done