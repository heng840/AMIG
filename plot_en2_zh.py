# plot Figure 3 + 4 from the paper -
# the decreasing ratio of the probability of the correct answer after suppressing knowledge neurons
import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import seaborn as sns


def load_and_aggregate_results(num_files):
    aggregated_results = defaultdict(lambda: defaultdict(int))

    for i in range(num_files):
        file_path = f'hallucination_temp_{i}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)

        for key in data:
            for relation, value in data[key].items():
                aggregated_results[key][relation] += value

    return aggregated_results
def format_data(results_data, key='suppression'):
    formatted = {}
    for uuid, data in results_data.items():
        if formatted.get(data["relation_name"]) is None:
            formatted[data["relation_name"]] = {"related": [], "unrelated": []}

        related_data = data[key]["related"]
        related_change = []
        for prob in related_data["pct_change"]:
            related_change.append(prob)

        unrelated_data = data[key]["unrelated"]
        unrelated_change = []
        for prob in unrelated_data["pct_change"]:
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
        pandas_format['related'].append(f"{verb} knowledge neurons for related facts")

        pandas_format['relation_name'].append(relation_name)
        pandas_format['pct_change'].append(data['unrelated'])
        pandas_format['related'].append(f"{verb} knowledge neurons for unrelated facts")
    return pd.DataFrame(pandas_format).dropna()


def plot_data(pd_df, experiment_type, out_path='test.png'):
    sns.set_theme(style="whitegrid")
    if experiment_type == "suppression":
        title = "Suppressing knowledge neurons"
    elif experiment_type == "enhancement":
        title = "Enhancing knowledge neurons"
    else:
        raise ValueError
    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=pd_df, kind="bar",
        x="relation_name", y="pct_change", hue="related",
        errorbar="sd", palette="dark", alpha=.6, height=6, aspect=4
    )
    g.despine(left=True)
    g.set_axis_labels("relation name", "Correct probability percentage change")
    g.legend.set_title(title)
    g.savefig(out_path)


def plot_from_json(results_json, edit_type):
    fig_dir = os.path.dirname(results_json)
    fig_dir = os.path.join(fig_dir, edit_type)
    os.makedirs(fig_dir, exist_ok=True)
    results = {}
    with open(results_json) as f:
        results.update(json.load(f))

    # plot results of suppression experiment
    suppression_data = format_data(results, key='suppression')
    enhancement_data = format_data(results, key='enhancement')
    plot_data(suppression_data, "suppression", out_path=f"{fig_dir}/suppress.png")

    # plot results of enhancement experiment
    plot_data(enhancement_data, "enhancement", out_path=f"{fig_dir}/enhance.png")


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser('Arguments for pararel result plotting')
    parser.add_argument('--en2zh_json',
                        default='kn_extend/7_13_with_synergistic/mbert/cross_language_results_0.2_0.3/en2zh_0.json',
                        type=str,)
    parser.add_argument('--zh2en_json',
                        default='kn_extend/7_13_with_synergistic/mbert/cross_language_results_0.2_0.3/zh2en_0.json',
                        type=str,)
    args = parser.parse_args()
    plot_from_json(results_json=args.en2zh_json, edit_type='en2zh')
    plot_from_json(results_json=args.zh2en_json, edit_type='zh2en')

