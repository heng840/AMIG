import argparse
import os
from pathlib import Path
from collections import Counter
import json

import pandas as pd
import seaborn as sns

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

font_path = 'times new roman.ttf'
font_properties = FontProperties(fname=font_path)


titles = {
    'mbert/en_threshold_0.3_0.2to0.25': 'm-Bert + English-DKN',
    'mbert/zh_threshold_0.2_0.2to0.25': 'm-Bert + Chinese-DKN',
    'mgpt/en_threshold_0.3_0.2to0.25': 'm-GPT + English-DKN',
    'mgpt/zh_threshold_0.2_0.2to0.25': 'm-GPT + Chinese-DKN',
    'mbert/cross_lingual': 'm-Bert + LIKN',
    'mgpt/cross_lingual': 'm-GPT + LIKN',  # C-dis
    'bert_0.3_0.2to0.25': 'Bert + DKN',
    'gpt_0.3_0.2to0.25': 'GPT-2 + DKN',  #ablation: k-dis and D-dis
}


def format_data(kn_dir, uuids_drop):
    kn_bag_counter = Counter()
    tot_kneurons = 0
    if isinstance(kn_dir, str):
        files_to_read = [kn_dir]
    else:
        files_to_read = kn_dir
    for filename in files_to_read:
        with open(filename) as f:
            kn_bag_dict = json.load(f)
            for uuid in kn_bag_dict:
                if uuid in uuids_drop:
                    continue
                kn_bag = kn_bag_dict[uuid]
                if 'synergistic_neurons_this_uuid' not in kn_bag and 'neurons' in kn_bag:
                    for kn in kn_bag['neurons']:
                        kn_bag_counter.update([kn[0]])
                        tot_kneurons += 1
                else:
                    if 'synergistic_neurons_this_uuid' in kn_bag:
                        for group in kn_bag['synergistic_neurons_this_uuid']:
                            for kn in group:
                                kn_bag_counter.update([kn[0]])
                                tot_kneurons += 1
    for k, v in kn_bag_counter.items():
        kn_bag_counter[k] = (v / tot_kneurons) * 100  # convert to percentage

    # Prepare DataFrame for Seaborn
    pd_df = pd.DataFrame({
        'Layer': list(kn_bag_counter.keys()),
        'Percentage': list(kn_bag_counter.values())
    })
    return pd_df


if __name__ == "__main__":
    def histogram():
        parser = argparse.ArgumentParser('Arguments for parallel result plotting')
        parser.add_argument('--base_dir',
                            # default='723final',
                            default='723',
                            type=str, )
        args = parser.parse_args()

        cases = [
            # 'mbert/en_threshold_0.3_0.2to0.25',
            # 'mbert/zh_threshold_0.2_0.2to0.25',
            # 'mgpt/en_threshold_0.3_0.2to0.25',
            # 'mgpt/zh_threshold_0.2_0.2to0.25',  # k-distribute and D-dis
            # 'mbert/cross_lingual',
            # 'mgpt/cross_lingual',  # C-dis
            'bert_0.3_0.2to0.25',
            'gpt_0.3_0.2to0.25',  #mono: D-dis
        ]
        combined_df = pd.DataFrame()

        for i, case in enumerate(cases):
            results_dir = os.path.join(args.base_dir, case)
            fig_dir = os.path.join(results_dir, 'figs')
            os.makedirs(fig_dir, exist_ok=True)

            if os.path.exists(os.path.join(results_dir, 'uuids_drop.json')):
                with open(os.path.join(results_dir, 'uuids_drop.json')) as f:
                    uuids_drop = json.load(f)
            else:
                uuids_drop = []

            if 'cross' in case:
                neuron_paths = Path(results_dir).glob('*cross*.json')  # C
            else:
                # neuron_paths = Path(results_dir).glob('*pararel_neurons_*.json')  # k
                neuron_paths = Path(results_dir).glob('*pararel_results_*.json')  # D
            # plot_kn_distribution(kn_dir=neuron_paths, uuids_drop=uuids_drop, ax=ax)
            pd_df = format_data(kn_dir=neuron_paths, uuids_drop=uuids_drop)

            # Add Model column based on the case and append to combined DataFrame
            pd_df['Model'] = titles[case]
            combined_df = combined_df._append(pd_df)
        plt.figure(figsize=(10, 6))  # Adjusted plot size
        palette = sns.color_palette("husl", 3)  # 2 colors from the "husl" palette
        palette3 = sns.color_palette("tab10", 3)  # for main D
        palette5 = sns.husl_palette(2, h=0.3)  # for main K
        sns.barplot(
            x='Layer',
            y='Percentage', hue='Model', data=combined_df, palette=palette)
        plt.xlabel(
            'Layer',
            fontsize=30, fontproperties=font_properties)  # Apply font to X label
        plt.ylabel('Percentage', fontsize=30, fontproperties=font_properties)
        legend = plt.legend(title='', fontsize=30, title_fontsize=30, loc='upper left')

        for text in legend.get_texts():
            text.set_fontproperties(font_properties)
            text.set_fontsize(30)  # Set the desired font size

        # Set the font properties for the legend title (if you have one)
        legend.get_title().set_fontproperties(font_properties)
        legend.get_title().set_fontsize(30)  # Set the desired font size

        # Set font properties for tick labels
        tick_font_properties = FontProperties(fname=font_path, size=28)

        plt.xticks(fontproperties=tick_font_properties)  # Apply font and size to X tick labels
        plt.yticks(fontproperties=tick_font_properties)  # Apply font and size to Y tick labels
        locations, labels = plt.xticks()
        plt.xticks(ticks=locations[::2], labels=labels[::2], fontproperties=tick_font_properties)
        # plt.xticks(ticks=locations[::4], labels=labels[::4], fontproperties=tick_font_properties)  # mgpt
        plt.tight_layout()  # Automatically adjust subplot parameters
        plt.show()

    histogram()
    # heatmap()
