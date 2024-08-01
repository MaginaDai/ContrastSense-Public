#! /bin/bash

working_directory=./runs # change working directory
dataset_path=../dataset # change path to dataset files

exp_name=collossl_single_run
train_device=right_pocket
eval_device=right_pocket
dataset_name=Shoaib_collossl.dat

# s=10

# python3 ../contrastive_training.py  -version "shot0" -shot ${s} --store "v0" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot1" -shot ${s} --store "v1" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot2" -shot ${s} --store "v2" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot3" -shot ${s} --store "v3" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot4" -shot ${s} --store "v4" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1



# for s in 1 5 50
# do

# python3 ../contrastive_training.py  -version "shot0" -ft_version "shot0" -ft True -shot ${s} --store "v0" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot1" -ft_version "shot1" -ft True -shot ${s} --store "v1" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot2" -ft_version "shot2" -ft True -shot ${s} --store "v2" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot3" -ft_version "shot3" -ft True -shot ${s} --store "v3" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "shot4" -ft_version "shot4" -ft True -shot ${s} --store "v4" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# done

s=10

# for alpha in 45 65
# do
# python3 ../contrastive_training.py  -version "alpha${alpha}_shot0" -shot ${s} --store "alpha${alpha}_shot0" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "alpha${alpha}_shot1" -shot ${s} --store "alpha${alpha}_shot1" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "alpha${alpha}_shot2" -shot ${s} --store "alpha${alpha}_shot2" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "alpha${alpha}_shot3" -shot ${s} --store "alpha${alpha}_shot3" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py  -version "alpha${alpha}_shot4" -shot ${s} --store "alpha${alpha}_shot4" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1
# done

# s=50
# for p in 60 80 100
# do

# python3 ../contrastive_training.py -version "shot0" -ft_version "tune_portion_${p}_shot0" -ft True -shot ${s} --pt_store "v0" --store "tune_portion_${p}_shot0" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py -version "shot1" -ft_version "tune_portion_${p}_shot1" -ft True -shot ${s} --pt_store "v1" --store "tune_portion_${p}_shot1" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py -version "shot2" -ft_version "tune_portion_${p}_shot2" -ft True -shot ${s} --pt_store "v2" --store "tune_portion_${p}_shot2" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py -version "shot3" -ft_version "tune_portion_${p}_shot3" -ft True -shot ${s} --pt_store "v3" --store "tune_portion_${p}_shot3" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# python3 ../contrastive_training.py -version "shot4" -ft_version "tune_portion_${p}_shot4" -ft True -shot ${s} --pt_store "v4" --store "tune_portion_${p}_shot4" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
# --multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
# --positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
# --learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

# done



python3 ../contrastive_training.py  -version "leave_shot0" -shot ${s} --store "leave_shot0" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

python3 ../contrastive_training.py  -version "leave_shot1" -shot ${s} --store "leave_shot1" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

python3 ../contrastive_training.py  -version "leave_shot2" -shot ${s} --store "leave_shot2" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

python3 ../contrastive_training.py  -version "leave_shot3" -shot ${s} --store "leave_shot3" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1

python3 ../contrastive_training.py  -version "leave_shot4" -shot ${s} --store "leave_shot4" --exp_name=${exp_name} --working_directory ${working_directory} --dataset_path ${dataset_path} --gpu_device=0 --training_mode=multi --training_epochs=100 \
--multi_sampling_mode=unsync_neg --held_out=0 --held_out_num_groups=5 --device_selection_strategy=closest_pos_all_neg --weighted_collossl --training_batch_size=512 \
--positive_devices --negative_devices --train_device=${train_device} --eval_device=${eval_device} --contrastive_temperature=0.05 --fine_tune_epochs=100 --eval_mode=base_model \
--learning_rate=1e-4 --dataset_name=${dataset_name} --neg_sample_size=1 --dynamic_device_selection=1
