# 8张图：KN
#python plot_pararel_results.py --results_dir 723final/mgpt/en_threshold_0.3_0.2to0.25
#python plot_pararel_results.py --results_dir 723final/mgpt/zh_threshold_0.2_0.2to0.25
#python plot_pararel_results.py --results_dir 723final/mbert/en_threshold_0.3_0.2to0.25
#python plot_pararel_results.py --results_dir 723final/mbert/zh_threshold_0.2_0.2to0.25
python plot_pararel_results.py --results_dir 723final/ablation/mgpt/en_threshold_0.3
python plot_pararel_results.py --results_dir 723final/ablation/mgpt/zh_threshold_0.2_725
python plot_pararel_results.py --results_dir 723final/ablation/mbert/en_threshold_0.3
python plot_pararel_results.py --results_dir 723final/ablation/mbert/zh_threshold_0.2
# 跨语言编辑
# 第一个对比使用：编辑一种语言 8 figs
#python plot_pararel_results.py --results_dir 723final/mgpt/cross_lingual --specific_filename en2zh_0.json
#python plot_pararel_results.py --results_dir 723final/mgpt/cross_lingual --specific_filename zh2en_0.json
#python plot_pararel_results.py --results_dir 723final/mbert/cross_lingual --specific_filename en2zh_0.json
#python plot_pararel_results.py --results_dir 723final/mbert/cross_lingual --specific_filename zh2en_0.json
## LIKN 8 figs
#python plot_pararel_results.py --results_dir 723final/mgpt/cross_lingual --specific_filename en_res.json
#python plot_pararel_results.py --results_dir 723final/mgpt/cross_lingual --specific_filename zh_res.json
#python plot_pararel_results.py --results_dir 723final/mbert/cross_lingual --specific_filename en_res.json
#python plot_pararel_results.py --results_dir 723final/mbert/cross_lingual --specific_filename zh_res.json
## seq 8figs
#python plot_pararel_results.py --results_dir 723final/mgpt/all_language --specific_filename en_res.json
#python plot_pararel_results.py --results_dir 723final/mgpt/all_language --specific_filename zh_res.json
#python plot_pararel_results.py --results_dir 723final/mbert/all_language --specific_filename en_res.json
#python plot_pararel_results.py --results_dir 723final/mbert/all_language --specific_filename zh_res.json