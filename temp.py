import json
from collections import Counter

# Load the data from the input file
with open('kn_extend/723/bert_0.3_0.2to0.25/pararel_results_0.json', 'r') as f:
    data = json.load(f)

# Filter out the unwanted properties
filtered_data = {}
for key, value in data.items():
    filtered_data[key] = {
        'neurons': value['neurons'],
        'relation_name': value['relation_name']
    }

# Write the filtered data to the output file
with open('kn_extend/723/bert_0.3_0.2to0.25/pararel_neurons_0.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

def format_data_hot(kn_dir, uuids_drop):
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
                if 'neurons' in kn_bag and 'synergistic_neurons_this_uuid' not in kn_bag:
                    kn_bag = kn_bag['neurons']
                    for kn in kn_bag:
                        # update counter with tuple of (layer_idx, neurons_idx)
                        kn_bag_counter.update([(kn[0], kn[1])])
                        tot_kneurons += 1
                else:
                    for group in kn_bag['synergistic_neurons_this_uuid']:
                        for kn in group:
                            kn_bag_counter.update([(kn[0], kn[1])])
                            tot_kneurons += 1

    # Prepare DataFrame for Seaborn
    layers = []
    neurons = []
    percentages = []
    for (layer, neuron), v in kn_bag_counter.items():
        percentage = (v / tot_kneurons) * 100 # convert to percentage
        layers.append(layer)
        neurons.append(neuron)
        percentages.append(percentage)

    pd_df = pd.DataFrame({
        'Layer': layers,
        'Neuron': neurons,
        'Percentage': percentages
    })
    return pd_df


def plot_kn_distribution(kn_dir, uuids_drop, ax):
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
    print(pd_df)
    # Draw a bar plot
    ax = sns.barplot(data=pd_df, x="Layer", y="Percentage", color="b", errorbar=None, ax=ax)
    ax.tick_params(labelsize=18)
    ax.set_xlabel('Layer', fontsize=24)
    ax.set_ylabel('Percentage', fontsize=24)


def plot_kn_distribution_hot(kn_dir, uuids_drop, ax, show_colorbar=False):
    kn_bag_counter = Counter()
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
                if 'neurons' in kn_bag and 'synergistic_neurons_this_uuid' not in kn_bag:
                    kn_bag = kn_bag['neurons']
                    for kn in kn_bag:
                        # update counter with tuple of (layer_idx, neurons_idx)
                        kn_bag_counter.update([(kn[0], kn[1])])
                else:
                    for group in kn_bag['synergistic_neurons_this_uuid']:
                        for kn in group:
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

    im = ax.scatter(layer_indices, neuron_indices, c=percentages, cmap='hot', norm=LogNorm())
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Percentage', fontsize=36)  # Set fontsize of colorbar label
        cbar.ax.yaxis.label.set_size(24)  # Set fontsize of colorbar tick labels
        cbar.ax.tick_params(labelsize=24)
    ax.set_xlabel('Layer Index', fontsize=24)
    ax.set_ylabel('Neuron Index', fontsize=24)
    ax.grid(True)

    def heatmap():
        base_dir = '723final'
        # case = 'mbert/en_threshold_0.3_0.2to0.25'
        case = 'mgpt/en_threshold_0.3_0.2to0.25'
        # for i, case in enumerate(cases):
        results_dir = os.path.join(base_dir, case)
        fig_dir = os.path.join(results_dir, 'figs')
        os.makedirs(fig_dir, exist_ok=True)

        # Load the uuids_drop list from the corresponding JSON file
        if os.path.exists(os.path.join(results_dir, 'uuids_drop.json')):
            with open(os.path.join(results_dir, 'uuids_drop.json')) as f:
                uuids_drop = json.load(f)
        else:
            uuids_drop = []

        neuron_paths = Path(results_dir).glob('*pararel_neurons_*.json')  # k
        pd_df = format_data_hot(kn_dir=neuron_paths, uuids_drop=uuids_drop)

        # Add Model column based on the case and append to combined DataFrame
        # pd_df['Model'] = titles[case]
        # combined_df = combined_df._append(pd_df)
        # Pivot the DataFrame to get it in the correct format for a heatmap
        heatmap_data = pd_df.pivot_table(index='Neuron', columns='Layer', values='Percentage')

        # Use logarithmic normalization
        norm = colors.LogNorm(vmin=heatmap_data.min().min() + 1e-10, vmax=heatmap_data.max().max())

        # Draw the heatmap
        plt.figure(figsize=(12, 8))
        font_size_caption = 30
        font_size_int = 28
        # Set the desired font size for colorbar here

        # Draw the heatmap and capture the colorbar axis
        ax = sns.heatmap(heatmap_data, cmap="YlGnBu", yticklabels=int(len(heatmap_data.index) / 10),
                         annot=False, norm=norm,
                         # cbar_kws={'ticks': [1, 0.1, 0.01, 0.001]}
                         )  # You can set your own tick values

        # Modify the colorbar tick labels
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size_int)

        # y_labels and x_labels
        y_labels = [i for i in range(0, len(heatmap_data.index), 1200)]  # for bert
        x_labels = [i for i in range(0, len(heatmap_data.columns), len(heatmap_data.columns) // 8)]  # for bert
        plt.yticks(y_labels, y_labels, rotation=0, fontsize=font_size_int)
        plt.xticks(x_labels, x_labels, fontsize=font_size_int)

        # Set the title and labels with custom font size
        plt.xlabel('Layer', fontsize=font_size_caption)
        # plt.ylabel('Neuron Index', fontsize=font_size_caption)
        plt.ylabel('', fontsize=font_size_caption)

        # Optionally, you can change the font size of the tick labels
        plt.xticks(fontsize=font_size_int)
        plt.yticks(fontsize=font_size_int)
        plt.tight_layout()
        plt.show()