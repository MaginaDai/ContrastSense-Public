version="shot"
store="shot_train"
dataset=Shoaib  ## You may change the name to other datasets

### You may use different split version, number of shots to test the model performance under different cross domain scenarios
python main_transfer_penalty.py -name ${dataset} -version "${version}0"  --pretrained "${store}0/${dataset}" --store "${store}0" -shot 10 &