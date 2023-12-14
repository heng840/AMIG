export CUDA_VISIBLE_DEVICES=1

#mgpt
mgpt_model_name=ai-forever/mGPT
mgpt_en_result_dir=723final/mgpt/en_threshold_0.3_0.2to0.25
mgpt_zh_result_dir=723final/mgpt/zh_threshold_0.2_0.2to0.25
mgpt_cross_lingual_output=723final/mgpt/cross_lingual
mgpt_all_language_output=723final/mgpt/all_language

#python plot_pararel_results.py \
#--results_dir $mgpt_en_result_dir

output_file="output_hallucination_detection_gpt_en.txt"
#幻觉实验
python hallucination_detection.py \
--model_name $mgpt_model_name \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_en.json \
--with_synergistic_neurons_result_dir $mgpt_en_result_dir \
--score_path threshold_tune/score_mgpt_en.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1
#python hallucination_detection.py \
#--model_name $mgpt_model_name \
#--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_zh.json \
#--with_synergistic_neurons_result_dir $mgpt_zh_result_dir \
#--score_path threshold_tune/score_mgpt_zh.json \
#--threshold_filter_DN 0.3
#
##跨语言编辑
##获取语言无关神经元
#python get_cross_language_neurons.py \
#--dir_en $mgpt_en_result_dir \
#--dir_zh $mgpt_zh_result_dir \
#--output $mgpt_cross_lingual_output
##编辑语言无关神经元、顺序编辑两种语言的神经元
#python get_cross_language_acc.py \
#--model_name $mgpt_model_name \
#--cross_language_neurons_json $mgpt_cross_lingual_output/cross_language_neurons.json \
#--cross_language_result_dir $mgpt_cross_lingual_output \
#--en_neurons $mgpt_en_result_dir \
#--zh_neurons $mgpt_zh_result_dir \
#--all_language_result_dir $mgpt_all_language_output
##编译一种语言的神经元
#python compare_cross_language_acc.py \
#--model_name $mgpt_model_name \
#--en_neurons $mgpt_en_result_dir \
#--zh_neurons $mgpt_zh_result_dir \
#--result_dir $mgpt_cross_lingual_output
##画图和存表
#python plot_cross_language_result.py \
#--cross_language_results_dir $mgpt_cross_lingual_output \
#--all_language_result_dir $mgpt_all_language_output \
#--results_jsons $mgpt_all_language_output/res_0.json $mgpt_cross_lingual_output/zh2en_0.json $mgpt_cross_lingual_output/en2zh_0.json
#python plot_en2_zh.py \
#--en2zh_json $mgpt_cross_lingual_output/en2zh_0.json \
#--zh2en_json $mgpt_cross_lingual_output/zh2en_0.json

