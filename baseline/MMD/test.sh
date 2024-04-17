shot=50
for portion in 80
do
    version="users_devices_tune_portion_${portion}_shot"
    name="CM_users_devices_tune_portion_${portion}_"
    name2="FM_users_devices_tune_portion_${portion}_"
    # python main.py -g 2 -version "${version}0" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}0" &
    # python main.py -g 2 -version "${version}1" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}1" &
    # python main.py -g 2 -version "${version}2" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}2" &
    # python main.py -g 2 -version "${version}3" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}3" &
    python main.py -g 1 -version "${version}4" -shot ${shot} -cross "multiple" -m 'CM' -name HASC --store "${name}4"

done