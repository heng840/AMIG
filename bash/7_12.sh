#export CUDA_VISIBLE_DEVICES=3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
bert_model_name=bert-base-multilingual-cased
bert_result_dir=7_12_result/mbert

gpu_nums=10
#en
#python main_garns.py \

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/correspond_dataset/en.json \
--neurons_result_dir $bert_result_dir/en_threshold_0.3 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.3

#zh :
python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $bert_result_dir/zh_threshold_0.25 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.25

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $bert_result_dir/zh_threshold_0.2 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2

python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir $bert_result_dir/zh_threshold_0.3 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.3

gpt_model_name=ai-forever/mGPT
# mgpt, with redundant

#en
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/monolingual/en_prompt_pararel.json \
#--neurons_result_dir $result_dir/mgpt/garns_threshold/en \
#--baseline_vector_path None \
#--batch_size=$batch_size \
#--steps=$steps
##zh
#python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
#--model_name $gpt_model_name \
#--pararel_json datasets/monolingual/ezh_prompt_pararel.json \
#--neurons_result_dir $result_dir/mgpt/garns_threshold/zh \
#--baseline_vector_path None \
#--batch_size=$batch_size \
#--steps=$steps \
#--adaptive_threshold=$threshold_zh
