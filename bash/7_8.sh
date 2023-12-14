#为了对比单语言bert的神经元分布。因为方法已经改进，所以再运行一次。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
gpu_nums=10
bert_model_name=bert-base-multilingual-cased
result_dir=7_9_Result_garns_wo
batch_size=20
steps=20
threshold_zh=0.2
redundant_threshold_percent=0.15
# mbert, with redundant
# bert已经测试完毕。

#en
python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/monolingual/en_prompt_pararel.json \
--neurons_result_dir $result_dir/mbert/garns_threshold/en \
--baseline_vector_path None \
--batch_size=$batch_size \
--steps=$steps \
#--redundant_threshold_percent=$redundant_threshold_percent
#zh
python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/monolingual/zh_prompt_pararel.json \
--neurons_result_dir $result_dir/mbert/garns_threshold/zh \
--baseline_vector_path None \
--batch_size=$batch_size \
--steps=$steps \
--adaptive_threshold=$threshold_zh
#--redundant_threshold_percent=$redundant_threshold_percent

#zh:t=0.25
python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/monolingual/ezh_prompt_pararel.json \
--neurons_result_dir $result_dir/mbert/garns_threshold/zh \
--baseline_vector_path None \
--batch_size=$batch_size \
--steps=$steps \
--adaptive_threshold=0.25
#zh:t=0.3
python -m torch.distributed.launch --nproc_per_node=$gpu_nums main_garns.py \
--model_name $bert_model_name \
--pararel_json datasets/monolingual/ezh_prompt_pararel.json \
--neurons_result_dir $result_dir/mbert/garns_threshold/zh \
--baseline_vector_path None \
--batch_size=$batch_size \
--steps=$steps \
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
