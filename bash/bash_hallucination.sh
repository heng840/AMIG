export CUDA_VISIBLE_DEVICES=0
bert_model_name=bert-base-multilingual-cased

python hallucination_detection.py \
--model_name bert-base-multilingual-cased \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_en.json \
--with_synergistic_neurons_result_dir 723final/mbert/en_threshold_0.3_0.2to0.25


#
#python hallucination_detection.py \
#--model_name $bert_model_name \
#--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_zh.json \
#--with_synergistic_neurons_result_dir 7_13_with_synergistic/mbert/zh_threshold_0.2_0.2to0.25