export CUDA_VISIBLE_DEVICES=1,2,3,4,5,7,8,9
#gpt_result_dir_all_dataset=723final/mgpt
gpu_nums=8
#gpt_model_name=ai-forever/mGPT

#output_file="output.txt"

##--neurons_result_dir $gpt_result_dir_all_dataset/zh_threshold_0.2_0.2to0.25 \
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/correspond_dataset/zh.json \
#--neurons_result_dir test_temp \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--batch_size=2 \
#--steps=10 \
#--k_path=1 \
#> $output_file 2>&1
#
#export CUDA_VISIBLE_DEVICES=1

#mgpt
mgpt_model_name=ai-forever/mGPT
mgpt_zh_result_dir=723final/mgpt/zh_threshold_0.2_0.2to0.25


output_file="output_hallucination_detection_gpt_zh_dis2.txt"
#幻觉实验
python -m torch.distributed.launch --nproc_per_node=$gpu_nums hallucination_detection.py \
--model_name $mgpt_model_name \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_zh.json \
--with_synergistic_neurons_result_dir $mgpt_zh_result_dir \
--score_path threshold_tune/score_mgpt_zh.json \
--threshold_filter_DN 0.7 \
--steps 10 \
--batch_size 10 \
> $output_file 2>&1
