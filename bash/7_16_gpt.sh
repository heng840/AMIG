export CUDA_VISIBLE_DEVICES=0,1,2,3
gpt_result_dir_all_dataset=7_20_add_sample_with_k_path/mgpt_all_neurons
batch_size=20
steps=20
gpu_nums=4
gpt_model_name=ai-forever/mGPT
k_path=10
#en
#python main_garns.py \

#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $gpt_result_dir_all_dataset/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--batch_size=$batch_size \
#--steps=$steps \
#--k_path=$k_path

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
--steps=$steps \
--k_path=1



bert_model_name=bert-base-multilingual-cased
bert_result_dir=7_18_add_n_sample/mbert

#en
#"对于英文而言,似乎还是该用convert_token_to_id."

# 不必。encode函数是对convert_token_to_id的上位覆盖。所所以全部的实验都应该重新运行了，包括消融实验。一共8个，需要40天。
# 最少做5个，需要25天
#python main_garns.py \

#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $bert_model_name \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $bert_result_dir/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25
#
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $bert_model_name \
#--pararel_json datasets/correspond_dataset/zh.json \
#--neurons_result_dir $bert_result_dir/zh_threshold_0.2_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25