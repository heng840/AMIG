#export CUDA_VISIBLE_DEVICES=0,1,2,3
#gpt_result_dir_all_dataset=723final/mgpt
#batch_size=20
#steps=20
#gpu_nums=4
#gpt_model_name=ai-forever/mGPT
##en
##python main_garns.py \
##第一个是消融实验，因为上次没跑完，还剩一个。
#
#output_file="output.txt"
##python -m torch.distributed.launch --nproc_per_node=$gpu_nums ablation_wo_sig.py \
##--model_name $gpt_model_name \
##--pararel_json datasets/correspond_dataset/zh.json \
##--neurons_result_dir 723final/ablation/mgpt/zh_threshold_0.2_725 \
##--baseline_vector_path None \
##--k_percent=-1 \
##--adaptive_threshold=0.2 \
##--batch_size=5 \
##--steps=10 \
##--k_path=1 > $output_file 2>&1
#
#
##下面是正式的实验。带有synergistic neurons
#
#bert_result_dir=723final/mbert
#bert_model_name=bert-base-multilingual-cased
##en
#
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $bert_model_name \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $bert_result_dir/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--k_path=1 > $output_file 2>&1
#
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $bert_model_name \
#--pararel_json datasets/correspond_dataset/zh.json \
#--neurons_result_dir $bert_result_dir/zh_threshold_0.2_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--k_path=1 > $output_file 2>&1
#
#
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $gpt_result_dir_all_dataset/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--batch_size=10 \
#--steps=$steps \
#--k_path=1 > $output_file 2>&1
#
#
##python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
##--model_name $gpt_model_name \
##--pararel_json datasets/correspond_dataset/zh.json \
##--neurons_result_dir $gpt_result_dir_all_dataset/zh_threshold_0.2_0.2to0.25 \
##--baseline_vector_path None \
##--k_percent=-1 \
##--adaptive_threshold=0.2 \
##--synergistic_threshold_percent_low=0.2 \
##--synergistic_threshold_percent_high=0.25 \
##--batch_size=10 \
##--steps=10 \
##--k_path=1 > $output_file 2>&1
#
#
export CUDA_VISIBLE_DEVICES=2,3,1
gpt_result_dir_all_dataset=723final/mgpt
gpu_nums=3
gpt_model_name=ai-forever/mGPT

output_file="output_zh1.txt"

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $gpt_model_name \
--pararel_json datasets/correspond_dataset/zh1.json \
--neurons_result_dir $gpt_result_dir_all_dataset/zh1_threshold_0.2_0.2to0.25 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2 \
--synergistic_threshold_percent_low=0.2 \
--synergistic_threshold_percent_high=0.25 \
--batch_size=10 \
--steps=20 \
--k_path=1 \
> $output_file 2>&1


