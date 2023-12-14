import argparse
import os

import seaborn as sns
import pandas as pd
import json

from matplotlib import pyplot as plt


def format_data(data, key):
    formatted = {}
    for uuid, item in data.items():
        if item["relation_name"] not in formatted:
            formatted[item["relation_name"]] = {"english_related": [], "english_unrelated": [],
                                                "chinese_related": [], "chinese_unrelated": []}

        for lang in ["english", "chinese"]:
            for relation_type in ["related", "unrelated"]:
                pct_change = item[lang][key][relation_type]["pct_change"]
                if pct_change:
                    avg_pct_change = sum(pct_change) / len(pct_change)
                    formatted[item["relation_name"]][f"{lang}_{relation_type}"].append(avg_pct_change)

    pandas_format = {'relation_name': [], 'type': [], 'pct_change': []}
    for relation_name, data in formatted.items():
        for lang_type in ["english_related", "english_unrelated", "chinese_related", "chinese_unrelated"]:
            if data[lang_type]:
                avg_pct_change = sum(data[lang_type]) / len(data[lang_type])
                pandas_format['relation_name'].append(relation_name)
                pandas_format['type'].append(lang_type)
                pandas_format['pct_change'].append(avg_pct_change)

    return pd.DataFrame(pandas_format)


def plot_data(df, operation, out_path):
    sns.set_theme(style="whitegrid")
    title = f"{operation.capitalize()} knowledge neurons"
    g = sns.catplot(
        data=df, kind="bar",
        x="relation_name", y="pct_change", hue="type",
        errorbar="sd", palette="dark", alpha=.6, height=6, aspect=4
    )
    g.despine(left=True)
    g.set_axis_labels("relation name", "Average Correct Probability Percentage Change")
    g.legend.set_title(title)
    g.savefig(out_path)


# # Suppression plot
# df_suppression = format_data(cross_data, "suppression")
# plot_data(df_suppression, "suppression", out_path=f'{args.cross_language_results_dir}/suppression.png')
#
# # Enhancement plot
# df_enhancement = format_data(cross_data, "enhancement")
# plot_data(df_enhancement, "enhancement", out_path=f'{args.cross_language_results_dir}/enhancement.png')
#
# df_suppression = format_data(all_data, "suppression")
# plot_data(df_suppression, "suppression", out_path=f'{args.all_language_results_dir}/suppression.png')
#
# # Enhancement plot
# df_enhancement = format_data(all_data, "enhancement")
# plot_data(df_enhancement, "enhancement", out_path=f'{args.all_language_results_dir}/enhancement.png')
# df_suppression = format_data(all_data, "suppression")
# plot_data_log(df_suppression, "suppression", out_path=f'{args.all_language_results_dir}/suppression_log_y.png')
#
# # Enhancement plot
# df_enhancement = format_data(all_data, "enhancement")
# plot_data_log(df_enhancement, "enhancement", out_path=f'{args.all_language_results_dir}/enhancement_log_y.png')


def calculate_averages_cross_language(json_files):
    if isinstance(json_files, str):
        json_files = [json_files]

    # Define dictionaries to store sum and counts for calculating average later
    sums = {
        "english": {
            "suppression_related": 0,
            "suppression_unrelated": 0,
            "enhancement_related": 0,
            "enhancement_unrelated": 0,
        },
        "chinese": {
            "suppression_related": 0,
            "suppression_unrelated": 0,
            "enhancement_related": 0,
            "enhancement_unrelated": 0,
        }
    }
    counts = {
        "english": {
            "suppression_related": 0,
            "suppression_unrelated": 0,
            "enhancement_related": 0,
            "enhancement_unrelated": 0,
        },
        "chinese": {
            "suppression_related": 0,
            "suppression_unrelated": 0,
            "enhancement_related": 0,
            "enhancement_unrelated": 0,
        }
    }

    # Initialize averages dictionary to hold results for each json file
    averages = {json_file: {} for json_file in json_files}

    # Iterate through each JSON file
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Iterate through each experiment
        for experiment_id, experiment_data in data.items():
            # Iterate through each language
            for language in ["english", "chinese"]:
                # Extract necessary data and update sums and counts
                sums[language]["suppression_related"] += experiment_data[language]['suppression']['related']['pct_change'][0]
                counts[language]["suppression_related"] += 1

                sums[language]["suppression_unrelated"] += experiment_data[language]['suppression']['unrelated']['pct_change'][0]
                counts[language]["suppression_unrelated"] += 1

                sums[language]["enhancement_related"] += experiment_data[language]['enhancement']['related']['pct_change'][0]
                counts[language]["enhancement_related"] += 1

                sums[language]["enhancement_unrelated"] += experiment_data[language]['enhancement']['unrelated']['pct_change'][0]
                counts[language]["enhancement_unrelated"] += 1

        # Calculate average for each case
        averages[json_file] = {
            language: {
                key: (sum_val / counts[language][key] if counts[language][key] != 0 else 0)
                for key, sum_val in sums[language].items()
            }
            for language in ["english", "chinese"]
        }

        # Calculate success rates
        for language in ["english", "chinese"]:
            if averages[json_file][language]["suppression_unrelated"] != 0:
                averages[json_file][language]["suppression_success"] = averages[json_file][language]["suppression_related"] / averages[json_file][language]["suppression_unrelated"]
            else:
                averages[json_file][language]["suppression_success"] = 0 # or whatever you want to assign when "suppression_unrelated" is zero

            if averages[json_file][language]["enhancement_unrelated"] != 0:
                averages[json_file][language]["enhancement_success"] = averages[json_file][language]["enhancement_related"] / averages[json_file][language]["enhancement_unrelated"]
            else:
                averages[json_file][language]["enhancement_success"] = 0 # or whatever you want to assign when "enhancement_unrelated" is zero

    return averages


def save_to_csv(averages, csv_file_path):
    # Flatten the averages dictionary to a list of dictionaries, each representing a row in the CSV
    rows = []
    for filename, languages in averages.items():
        for language, metrics in languages.items():
            row = {'filename': filename, 'language': language}
            row.update(metrics)
            rows.append(row)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(rows)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)


parser = argparse.ArgumentParser('Arguments for pararel result plotting')
parser.add_argument('--cross_lingual',
                    default='723final/mbert/cross_lingual',
                    type=str, )
parser.add_argument('--all_language',
                    default='723final/mbert/all_language',
                    type=str, )
args = parser.parse_args()
results_list = [
    os.path.join(args.all_language, 'res_0.json'),
    os.path.join(args.cross_lingual, 'en2zh_0.json'),
    os.path.join(args.cross_lingual, 'zh2en_0.json'),
    os.path.join(args.cross_lingual, 'res_0.json'),
]
averages = calculate_averages_cross_language(results_list)
save_to_csv(averages, f'{args.all_language}/cross_language_averages.csv')
