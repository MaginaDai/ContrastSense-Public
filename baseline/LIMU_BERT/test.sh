version="shot"
name="user"
# 1 2 3 4
# 5 10 20 50
for shot in 50
do
    for v in 4
    do
        python classifier_bert.py v1_v2 "${version}${v}" -g 2 -f "${name}" -shot ${shot} -p HASC -name 'HASC' -s "HASC_ft_shot_${shot}" 

        wait

    done
done