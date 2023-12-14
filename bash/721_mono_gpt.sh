export CUDA_VISIBLE_DEVICES=3
k_path=1
gpt_result_dir=723/gpt_0.3_0.2to0.25

#python main_garns.py \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $gpt_result_dir \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.2 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25 \
#--k_path=$k_path

#前面的实验已经完成。下面做幻觉实验。
output_file='output_gpt_h.txt'
python hallucination_detection.py \
--model_name gpt2 \
--wrong_fact_dataset_json datasets/correspond_dataset/wrong_fact_en.json \
--with_synergistic_neurons_result_dir $gpt_result_dir \
--threshold 0.000001  \
--score_path threshold_tune/score_gpt2.json \
--threshold_filter_DN 0.7 \
> $output_file 2>&1