export CUDA_VISIBLE_DEVICES=0

#mbert  在137的0号卡上跑了。
mbert_model_name=bert-base-multilingual-cased
bert_zh_result_dir=723final/mbert/zh_threshold_0.2_0.2to0.25


output_file="output_hallucination_detection_bert_zh.txt"
#幻觉实验
python hallucination_detection.py \
--model_name $mbert_model_name \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_zh.json \
--with_synergistic_neurons_result_dir $bert_zh_result_dir \
--score_path threshold_tune/score_mbert_zh.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1
