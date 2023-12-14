export CUDA_VISIBLE_DEVICES=1

#mgpt
mgpt_model_name=ai-forever/mGPT
mgpt_zh_result_dir=723final/mgpt/zh_threshold_0.2_0.2to0.25


output_file="output_hallucination_detection_gpt_zh.txt"
#幻觉实验
python hallucination_detection.py \
--model_name $mgpt_model_name \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_zh.json \
--with_synergistic_neurons_result_dir $mgpt_zh_result_dir \
--score_path threshold_tune/score_mgpt_zh.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1