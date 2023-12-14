export CUDA_VISIBLE_DEVICES=3
bert_result_dir=723/bert_0.3_0.2to0.25
k_path=1
#en
#python main_garns.py \
#--model_name bert-base-cased \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $bert_result_dir \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--k_path=$k_path


output_file='output_bert_h.txt'
python hallucination_detection.py \
--model_name bert-base-cased \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_en.json \
--with_synergistic_neurons_result_dir $bert_result_dir \
--threshold 0.000001  \
--score_path threshold_tune/score_bert.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1
