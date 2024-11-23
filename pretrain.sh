version="shot"
store="shot_train"
### You may use different split version to test the model performance under different cross domain scenarios
python main.py  -name Shoaib -version "${version}0" --store "${store}0" -cross "users" -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels 0.1 -slr 0.7

### Uncomment to train models on other IMU datasets 
# python main.py  -name HHAR -version "${version}0" --store "${store}0" -cross "users" -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels 0.1 -slr 0.7
# python main.py  -name MotionSense -version "${version}0" --store "${store}0" -cross "users" -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels 0.1 -slr 0.7
# python main.py  -name HASC -version "${version}0" --store "${store}0" -cross "users" -hard True -time_window 60 -last_ratio 0.5 -label_type 1 -tem_labels 0.1 -slr 0.7


### Uncomment to train models on EMG datasets 