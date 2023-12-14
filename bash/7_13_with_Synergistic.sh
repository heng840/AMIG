#export CUDA_VISIBLE_DEVICES=3

export CUDA_VISIBLE_DEVICES=0,1,2,5,6,7,8,9
bert_model_name=bert-base-multilingual-cased
bert_result_dir=7_13_with_synergistic/mbert

gpu_nums=8
#en
#python main_garns.py \

#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $bert_model_name \
#--pararel_json datasets/correspond_dataset/en.json \
#--neurons_result_dir $bert_result_dir/en_threshold_0.3_0.2to0.25 \
#--baseline_vector_path None \
#--k_percent=-1 \
#--adaptive_threshold=0.3 \
#--synergistic_threshold_percent_low=0.2 \
#--synergistic_threshold_percent_high=0.25

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $bert_result_dir/zh_threshold_0.2_0.2to0.25 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2 \
--synergistic_threshold_percent_low=0.2 \
--synergistic_threshold_percent_high=0.25

gpt_model_name=ai-forever/mGPT
