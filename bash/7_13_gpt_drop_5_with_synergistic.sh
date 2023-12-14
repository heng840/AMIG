#export CUDA_VISIBLE_DEVICES=3
## 因为分布式训练被占用，先测试小数据集。
##7/16,.修改了gains,py的源代码。
gpt_model_name=ai-forever/mGPT
#gpt_result_dir=7_13_with_synergistic/mgpt
#
#python main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/en_P39_264.json \
#--neurons_result_dir $gpt_result_dir/test_small_dataset/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25
#
#python main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/zh_P39_264.json \
#--neurons_result_dir $gpt_result_dir/test_small_dataset/zh_threshold_0.2_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25


export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8,9
gpt_result_dir_all_dataset=7_16_with_synergistic/mgpt_all_neurons
batch_size=20
steps=20
gpu_nums=9
#en
#python main_garns.py \

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $gpt_model_name \
--pararel_json datasets/correspond_dataset/en.json \
--neurons_result_dir $gpt_result_dir_all_dataset/en_threshold_0.3_0.2to0.25 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.3 \
--synergistic_threshold_percent_low=0.2 \
--synergistic_threshold_percent_high=0.25 \
--batch_size=$batch_size \
--steps=$steps

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $gpt_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $gpt_result_dir_all_dataset/zh_threshold_0.2_0.2to0.25 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2 \
--synergistic_threshold_percent_low=0.2 \
--synergistic_threshold_percent_high=0.25 \
--batch_size=$batch_size \
--steps=$steps
