export CUDA_VISIBLE_DEVICES=2

#mbert
mbert_model_name=bert-base-multilingual-cased
bert_en_result_dir=723final/mbert/en_threshold_0.3_0.2to0.25

#python plot_pararel_results.py \
#--results_dir $bert_en_result_dir

output_file="output_hallucination_detection_bert_en.txt"
#幻觉实验
python hallucination_detection.py \
--model_name $mbert_model_name \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_en.json \
--with_synergistic_neurons_result_dir $bert_en_result_dir \
--score_path threshold_tune/score_mbert_en.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1