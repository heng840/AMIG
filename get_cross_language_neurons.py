import argparse
import json
import os
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import seaborn as sns
import pandas as pd

def find_language_independent_neurons(dir_en, dir_zh):
    en_data = {}
    zh_data = {}

    # Load English neuron data
    for file in os.listdir(dir_en):
        if 'neurons' in file:
            with open(os.path.join(dir_en, file), 'r') as f:
                en_data.update(json.load(f))

    # Load Chinese neuron data
    for file in os.listdir(dir_zh):
        if 'neurons' in file:
            with open(os.path.join(dir_zh, file), 'r') as f:
                zh_data.update(json.load(f))

    # Find common uuids
    common_uuids = set(en_data.keys()).intersection(set(zh_data.keys()))

    # Now find common neurons for each common uuid
    language_independent_neurons = {}
    for uuid in common_uuids:
        en_neurons = set(tuple(neuron) for neuron in en_data[uuid]['neurons'])
        zh_neurons = set(tuple(neuron) for neuron in zh_data[uuid]['neurons'])

        # Find common neurons
        common_neurons = en_neurons.intersection(zh_neurons)
        if common_neurons:
            language_independent_neurons[uuid] = {
                "neurons": [list(neuron) for neuron in common_neurons],
                "relation_name": en_data[uuid]['relation_name']  # assuming relation_name is same for a given uuid
            }

    # Return language-independent neurons
    return language_independent_neurons


def plot_kn_distribution(filename, fig_dir):
    kn_bag_counter = Counter()
    tot_kneurons = 0

    # loop over all JSON files in the directory
    with open(filename) as f:
        kn_bag_dict = json.load(f)
        for uuid in kn_bag_dict:
            # if uuid in uuids_drop:
            #     continue
            kn_bag = kn_bag_dict[uuid]
            if 'neurons' in kn_bag:
                for kn in kn_bag['neurons']:
                    kn_bag_counter.update([kn[0]])
                    tot_kneurons += 1

    for k, v in kn_bag_counter.items():
        kn_bag_counter[k] = (v / tot_kneurons) * 100  # convert to percentage

    # Prepare DataFrame for Seaborn
    pd_df = pd.DataFrame({
        'Layer': list(kn_bag_counter.keys()),
        'Percentage': list(kn_bag_counter.values())
    })

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Draw a bar plot
    ax = sns.barplot(data=pd_df, x="Layer", y="Percentage", color="b", errorbar=None)
    ax.set(title="Distribution of neurons across layers")

    plt.savefig(os.path.join(fig_dir, 'cross_lingual_neurons.png'), dpi=100)
    plt.close()


def plot_kn_distribution_hot(filename, fig_dir):
    kn_bag_counter = Counter()
    with open(filename) as f:
        kn_bag_dict = json.load(f)
        for uuid in kn_bag_dict:
            kn_bag = kn_bag_dict[uuid]
            if 'relation_name' in kn_bag:
                kn_bag = kn_bag['neurons']
            for kn in kn_bag:
                # update counter with tuple of (layer_idx, neurons_idx)
                kn_bag_counter.update([(kn[0], kn[1])])

    # create arrays for the layer indices, neuron indices, and counts
    layer_indices = []
    neuron_indices = []
    counts = []

    for (layer, neuron), count in kn_bag_counter.items():
        layer_indices.append(layer)
        neuron_indices.append(neuron)
        counts.append(count)

    # normalize the counts to get percentages
    percentages = np.array(counts) / sum(counts)

    plt.figure(figsize=(10, 10))
    plt.scatter(layer_indices, neuron_indices, c=percentages, cmap='hot', norm=LogNorm())
    plt.colorbar(label='Percentage')
    plt.xlabel('Layer Index')
    plt.ylabel('Neuron Index')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, 'hot_cross_lingual_neurons.png'), dpi=100)
    plt.close()



def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Find language independent neurons")
    parser.add_argument("--dir_en", help="Directory containing first set of neuron json files",
                        default='7_13_with_synergistic/mbert/zh_threshold_0.2_0.2to0.25',
                    )
    parser.add_argument("--dir_zh", help="Directory containing second set of neuron json files",
                         default='7_13_with_synergistic/mbert/en_threshold_0.3_0.2to0.25'
                        )
    parser.add_argument("--output",
                        default='723final/mbert/cross_lingual'
                        )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    # Find language-independent neurons
    neurons = find_language_independent_neurons(dir_en=args.dir_en, dir_zh=args.dir_zh)

    # Write to output file
    with open(f"{args.output}/cross_language_neurons.json", 'w') as f:
        json.dump(neurons, f)

    # plot_kn_distribution(filename=f"{args.output}/cross_language_neurons.json", fig_dir=args.output)
    # fixme 在plot_sub_res中统一进行。
    # plot_kn_distribution_hot(filename=f"{args.output}/cross_language_neurons.json", fig_dir=args.output)


if __name__ == "__main__":
    main()
