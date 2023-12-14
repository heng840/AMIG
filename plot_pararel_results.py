# plot Figure 3 + 4 from the paper -
# the decreasing ratio of the probability of the correct answer after suppressing knowledge neurons
import os
from collections import Counter
import json

import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import argparse

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties

uuids_drop = []
uuids_drop_unrelated = []
enhance_drop_t = 250
suppress_drop_t = 20
uuids_drop_suppress = []

font_path = 'times new roman.ttf'
font_properties = FontProperties(fname=font_path)
def format_data(results_data, key='suppression', dir_uuid=None):
    uuids_drop_dict = []
    formatted = {}
    for uuid, data in results_data.items():
        if formatted.get(data["relation_name"]) is None:
            formatted[data["relation_name"]] = {"related": [], "unrelated": []}

        related_data = data[key]["related"]
        related_change = []
        for prob in related_data["pct_change"]:
            if key == 'enhancement':
                if uuid not in uuids_drop and prob >= enhance_drop_t:
                    uuids_drop_dict.append({uuid: prob})
                    uuids_drop.append(uuid)
                    # print(uuid,prob)
                if prob < enhance_drop_t:
                    related_change.append(prob)
            else:
                if uuid not in uuids_drop and prob > suppress_drop_t:
                    uuids_drop.append(uuid)
                    uuids_drop_dict.append({uuid: prob})
                    # print(prob)
                if prob <= suppress_drop_t:
                    related_change.append(prob)

        unrelated_data = data[key]["unrelated"]
        unrelated_change = []
        for prob in unrelated_data["pct_change"]:
            if key == 'enhancement':
                if uuid not in uuids_drop and prob >= enhance_drop_t:
                    uuids_drop.append(uuid)
                    uuids_drop_dict.append({uuid: prob})
                    # print(uuid,prob)
                if prob < enhance_drop_t:
                    unrelated_change.append(prob)
            else:
                if uuid not in uuids_drop and prob > suppress_drop_t:
                    uuids_drop.append(uuid)
                    uuids_drop_dict.append({uuid: prob})
                    # print(prob)
                if prob <= suppress_drop_t:
                    unrelated_change.append(prob)
        if related_change:
            related_change = sum(related_change) / len(
                related_change
            )
            if unrelated_change:
                unrelated_change = sum(unrelated_change) / len(
                    unrelated_change
                )
            else:
                unrelated_change = 0.0
            formatted[data["relation_name"]]["related"].append(related_change)
            formatted[data["relation_name"]]["unrelated"].append(
                unrelated_change
            )

    # Save uuids_drop to a file
    with open(f'{dir_uuid}/uuids_drop.json', 'w') as f:
        json.dump(uuids_drop, f)
    for relation_name, data in formatted.items():
        if data["related"]:
            data["related"] = sum(data["related"]) / len(data["related"])
        else:
            data["related"] = float("nan")
        if data["unrelated"]:
            data["unrelated"] = sum(data["unrelated"]) / len(data["unrelated"])
        else:
            data["unrelated"] = float("nan")

    pandas_format = {'relation_name': [], 'related': [], 'pct_change': []}
    for relation_name, data in formatted.items():
        verb = "Suppressing" if key == "suppression" else "Enhancing"
        pandas_format['relation_name'].append(relation_name)
        pandas_format['pct_change'].append(data['related'])
        pandas_format['related'].append(f"for related facts")

        pandas_format['relation_name'].append(relation_name)
        pandas_format['pct_change'].append(data['unrelated'])
        pandas_format['related'].append(f"for unrelated facts")
    return pd.DataFrame(pandas_format).dropna()


def plot_data(pd_df, experiment_type, out_path="test.png", font_properties=None):
    sns.set_theme(style="whitegrid")
    if experiment_type == "suppression":
        title = "       Suppressing\n knowledge neurons"
    elif experiment_type == "enhancement":
        title = "       Enhancing\n knowledge neurons"
    else:
        raise ValueError
    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=pd_df,
        kind="bar",
        x="relation_name",
        y="pct_change",
        hue="related",
        errorbar="sd",
        palette="dark",
        alpha=0.6,
        height=6,
        aspect=4,
    )
    # Set the tick labels font properties with specified font size
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties, rotation=60,
                           fontsize=30)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties,
                           fontsize=30)
        ax.set_xlabel("Relation name", fontproperties=font_properties,
                      fontsize=36)
        ax.set_ylabel("Probability change", fontproperties=font_properties,
                      fontsize=36)  # Set y-axis label with desired font properties
    g.legend.set_title(title)
    legend_title = g.legend.get_title()
    legend_title.set_fontproperties(font_properties)
    legend_title.set_fontsize(36)
    g.legend.set_bbox_to_anchor((1.165, 0.5))
    plt.yticks(ticks=g.ax.get_yticks(), labels=[f'{val:.2f}' for val in g.ax.get_yticks()])
    for text in g.legend.texts:
        text.set_font_properties(font_properties)
        text.set_fontsize(36)

    plt.tight_layout()
    # plt.show()
    g.savefig(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Arguments for pararel result plotting')
    parser.add_argument('--results_dir',
                        # default='723final/mgpt/cross_lingual',
                        default='723final/mbert/cross_lingual',
                        type=str,
                        help='directory in which the results from pararel_evaluate.py are saved.')
    parser.add_argument('--specific_filename',
                        # default='723final/mgpt/cross_lingual',
                        default='zh2en_0.json',
                        type=str,
                        help='directory in which the results from pararel_evaluate.py are saved.')
    args = parser.parse_args()

    # Replace this filename with the specific JSON file you want to read
    specific_filename = args.specific_filename
    file_name_without_extension, _ = os.path.splitext(specific_filename)
    # Build the path to the specific JSON file
    specific_filepath = os.path.join(args.results_dir, specific_filename)

    # Open the file and read the results
    # with open(specific_filepath) as f:
    #     results = json.load(f)


    fig_dir = os.path.join(args.results_dir, 'figs_appendix')
    os.makedirs(fig_dir, exist_ok=True)
    results_dir = Path(args.results_dir)

    # load results
    result_paths = results_dir.glob('*pararel_results_*.json')
    results = {}
    for p in result_paths:
        with open(p) as f:
            results.update(json.load(f))

    # plot results of suppression experiment
    suppression_data = format_data(results, key='suppression', dir_uuid=args.results_dir)
    enhancement_data = format_data(results, key='enhancement', dir_uuid=args.results_dir)

    # plot_data(suppression_data, "suppression", out_path=f"{fig_dir}/{file_name_without_extension}_suppress.png", font_properties=font_properties)
    plot_data(suppression_data, "suppression", out_path=f"{fig_dir}/suppress.png", font_properties=font_properties)

    # plot results of enhancement experiment
    # plot_data(enhancement_data, "enhancement", out_path=f"{fig_dir}/{file_name_without_extension}_enhance.png", font_properties=font_properties)
    plot_data(enhancement_data, "enhancement", out_path=f"{fig_dir}/enhance.png", font_properties=font_properties)
    # Path to the Times New Roman font file
