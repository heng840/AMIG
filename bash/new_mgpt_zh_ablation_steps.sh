export CUDA_VISIBLE_DEVICES=0,3,8,9
gpu_nums=4
gpt_model_name=ai-forever/mGPT

output_file="output_steps20.txt"
python -m torch.distributed.launch --nproc_per_node=$gpu_nums ablation_wo_sig.py \
--model_name $gpt_model_name \
--pararel_json datasets/correspond_dataset/zh.json \
--neurons_result_dir 723final/ablation/mgpt/zh_threshold_0.2_steps20 \
--baseline_vector_path None \
--k_percent=-1 \
--adaptive_threshold=0.2 \
--batch_size=20 \
--steps=20 \
--k_path=1 > $output_file 2>&1