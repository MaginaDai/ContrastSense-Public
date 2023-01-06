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

version="shot"
e=400
ewc=1
lr=0.0005
v=2
for k in 512
do 
    name="MoCo_K${k}_v${v}"

    python main.py -g 3 -moco_K ${k} -label_type 1 -version "shot${v}" -name HASC --store "${name}" &
    python main.py -g 3 -moco_K ${k} -label_type 1 -version "shot${v}" -name HHAR --store "${name}" &
    python main.py -g 2 -moco_K ${k} -label_type 1 -version "shot${v}" -name MotionSense --store "${name}" &
    python main.py -g 2 -moco_K ${k} -label_type 1 -version "shot${v}" -name Shoaib --store "${name}"

    wait
    
    store="MoCo_K${k}_v${v}"

    python main_trans_ewc.py -g 2 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HASC --pretrained "${name}/HASC" --store ${store} &
    python main_trans_ewc.py -g 2 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name HHAR --pretrained "${name}/HHAR" --store ${store} &
    python main_trans_ewc.py -g 3 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name Shoaib --pretrained "${name}/Shoaib" --store ${store} &
    python main_trans_ewc.py -g 3 -moco_K ${k} -e ${e} -ewc_lambda ${ewc} -ewc True -ft True -lr ${lr} -version "shot${v}" -shot 10 -name MotionSense  --pretrained "${name}/MotionSense" --store ${store} 

    wait
done
