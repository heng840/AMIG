export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9
gpt_result_dir_all_dataset=723final/mgpt
gpu_nums=8
gpt_model_name=ai-forever/mGPT
#en
#python main_garns.py \
#第一个是消融实验，因为上次没跑完，还剩一个。
#
#output_file="output_steps20.txt"
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums ablation_wo_sig.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/zh.json \
#--neurons_result_dir 723final/ablation/mgpt/zh_threshold_0.2_steps20 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--batch_size=20 \
#--steps=20 \
#--k_path=1 > $output_file 2>&1

output_file1="output_zh_steps20.txt"

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $gpt_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $gpt_result_dir_all_dataset/zh_threshold_0.2_0.2to0.25steps20 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2 \
--synergistic_threshold_percent_low=0.2 \
--synergistic_threshold_percent_high=0.25 \
--batch_size=20 \
--steps=20 \
--k_path=1 \
> $output_file1 2>&1


