import os
import json
import argparse
import pandas as pd

enhance_drop_t = 250
suppress_drop_t = 20  # 最后选好了阈值。
def calculate_averages(results_dir):
    # List all files in the directory
    if os.path.isdir(results_dir):
        files = os.listdir(results_dir)
        # Filter only .json files and those containing "results"
        json_files = [f for f in files if f.endswith('.json') and 'pararel_results' in f]
    else:
        json_files = [results_dir]
    # Define dictionaries to store sum and counts for calculating average later
    sums = {
        "suppression_related": 0,
        "suppression_unrelated": 0,
        "enhancement_related": 0,
        "enhancement_unrelated": 0,
    }
    counts = {
        "suppression_related": 0,
        "suppression_unrelated": 0,
        "enhancement_related": 0,
        "enhancement_unrelated": 0,
    }

    # Iterate through each JSON file
    confusion_s = 0
    confusion_e = 0
    for file in json_files:
        if os.path.isdir(results_dir):
            filename = os.path.join(results_dir, file)
        else:
            filename = file
        with open(filename) as f:
            data = json.load(f)

        # Iterate through each experiment
        for experiment_id, experiment_data in data.items():
            # Extract necessary data and update sums and counts
            if experiment_data['suppression']['related']['pct_change'][0] <= suppress_drop_t:
                sums["suppression_related"] += experiment_data['suppression']['related']['pct_change'][0]
                counts["suppression_related"] += 1
            else:
                confusion_s += 1
            if experiment_data['suppression']['unrelated']['pct_change'][0] <= suppress_drop_t:
                sums["suppression_unrelated"] += experiment_data['suppression']['unrelated']['pct_change'][0]
                counts["suppression_unrelated"] += 1
            if experiment_data['enhancement']['related']['pct_change'][0] < enhance_drop_t:
                sums["enhancement_related"] += experiment_data['enhancement']['related']['pct_change'][0]
                counts["enhancement_related"] += 1
            else:
                confusion_e += 1
            if experiment_data['enhancement']['unrelated']['pct_change'][0] < enhance_drop_t:
                sums["enhancement_unrelated"] += experiment_data['enhancement']['unrelated']['pct_change'][0]
                counts["enhancement_unrelated"] += 1

    # Calculate average for each case
    averages = {key: (sum_val / counts[key] if counts[key] != 0 else 0) for key, sum_val in sums.items()}
    print(confusion_e/len(data))
    print(confusion_s/len(data))
    if averages["suppression_unrelated"] != 0:
        averages["suppression_success"] = averages["suppression_related"] / averages["suppression_unrelated"]
    else:
        averages["suppression_success"] = 0  # or whatever you want to assign when "suppression_unrelated" is zero

    if averages["enhancement_unrelated"] != 0:
        averages["enhancement_success"] = averages["enhancement_related"] / averages["enhancement_unrelated"]
    else:
        averages["enhancement_success"] = 0  # or whatever you want to assign when "enhancement_unrelated" is zero
    averages['Succ'] = abs(averages["enhancement_success"]) + abs(averages["suppression_success"])
    averages["directory"] = results_dir
    return averages


def convert_lang2_json(json_file):
    # Load the original JSON data
    with open(json_file, "r") as file:
        data = json.load(file)

    # Split the data into English and Chinese data
    data_en = {k: v['english'] for k, v in data.items()}
    data_zh = {k: v['chinese'] for k, v in data.items()}

    # Add relation_name to each item
    for k, v in data.items():
        if 'relation_name' in v:
            data_en[k]['relation_name'] = v['relation_name']
            data_zh[k]['relation_name'] = v['relation_name']

    # Write the English data to a new JSON file
    with open(f'{os.path.dirname(json_file)}/en_res.json', "w") as file:
        json.dump(data_en, file)

    # Write the Chinese data to a new JSON file
    with open(f'{os.path.dirname(json_file)}/zh_res.json', "w") as file:
        json.dump(data_zh, file)

def main():
    # Use argparse for command-line arguments
    parser = argparse.ArgumentParser(description='Calculate averages from results directories.')
    parser.add_argument('--results_dir', type=str, nargs='+',
                        default=[
                            '723final/mbert/en_threshold_0.3_0.2to0.25',
                            '723final/ablation/mbert/en_threshold_0.3',  # baseline

                            '723final/mbert/zh_threshold_0.2_0.2to0.25',
                            '723final/ablation/mbert/zh_threshold_0.2',

                            '723final/mgpt/zh_threshold_0.2_0.2to0.25',
                            '723final/ablation/mgpt/zh_threshold_0.2_725',

                            '723final/mgpt/en_threshold_0.3_0.2to0.25',
                            '723final/ablation/mgpt/en_threshold_0.3',
                                 ],
                        help='Paths to the results directories')
    parser.add_argument('--output_file', type=str, default='723final/ablation/csv',
                        help='Path to the output file')

    parser.add_argument('--cross_lingual_bert',
                        default='723final/mbert/cross_lingual',
                        type=str, )
    parser.add_argument('--all_language_bert',
                        default='723final/mbert/all_language',
                        type=str, )
    parser.add_argument('--cross_lingual_gpt',
                        default='723final/mgpt/cross_lingual',
                        type=str, )
    parser.add_argument('--all_language_gpt',
                        default='723final/mgpt/all_language',
                        type=str, )
    args = parser.parse_args()

    convert_lang2_json(os.path.join(args.all_language_gpt, 'res_0.json'))
    convert_lang2_json(os.path.join(args.cross_lingual_gpt, 'res_0.json'))
    cross_lingual_results_list = [
        os.path.join(args.all_language_bert, 'en_res.json'),
        os.path.join(args.all_language_bert, 'zh_res.json'),
        os.path.join(args.cross_lingual_bert, 'en2zh_0.json'),
        os.path.join(args.cross_lingual_bert, 'zh2en_0.json'),
        os.path.join(args.cross_lingual_bert, 'en_res.json'),
        os.path.join(args.cross_lingual_bert, 'zh_res.json'),
        os.path.join(args.all_language_gpt, 'en_res.json'),
        os.path.join(args.all_language_gpt, 'zh_res.json'),
        os.path.join(args.cross_lingual_gpt, 'en2zh_0.json'),
        os.path.join(args.cross_lingual_gpt, 'zh2en_0.json'),
        os.path.join(args.cross_lingual_gpt, 'en_res.json'),
        os.path.join(args.cross_lingual_gpt, 'zh_res.json'),
    ]
    os.makedirs(args.output_file, exist_ok=True)
    # Create a DataFrame to store results from all directories
    df = pd.DataFrame()

    for directory in args.results_dir:  # 消融实验
    # for directory in cross_lingual_results_list:  # 跨语言实验
        averages = calculate_averages(directory)
        averages_df = pd.DataFrame(averages, index=[0])  # Convert dictionary to DataFrame
        df = df._append(averages_df, ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(args.output_file, 'ablation.csv'), index=False)  # 消融实验
    # df.to_csv(os.path.join(os.path.dirname(args.all_language_gpt), 'cross_lingual_edit.csv'), index=False)


if __name__ == '__main__':
    main()
