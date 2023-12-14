import itertools


def test_redundant_neurons(prompt, ground_truth, all_neurons, T):
    redundant_neurons = {}
    all_subsets = []
    for subset_size in range(1, len(all_neurons) + 1):
        for subset in itertools.combinations(all_neurons, subset_size):
            all_subsets.append(list(subset))

    original_results, _ = prediction_accuracy(prompt, ground_truth, [])
    original_accuracy = original_results["after"]["gt_prob"]

    for subset in all_subsets:
        neurons_to_suppress = [neuron for neuron in all_neurons if neuron not in subset]
        new_results, _ = prediction_accuracy(prompt, ground_truth, neurons_to_suppress, undo_modification=True)
        new_accuracy = new_results["after"]["gt_prob"]
        if new_accuracy >= original_accuracy - T:
            if len(subset) in redundant_neurons:
                redundant_neurons[len(subset)].append(subset)
            else:
                redundant_neurons[len(subset)] = [subset]

    return redundant_neurons


def calculate_metrics(redundant_neurons, total_neurons):
    # Prepare a set to store all redundant neurons
    redundant_neurons_set = set()

    # Count all unique redundant neurons
    for _, subsets in redundant_neurons.items():
        for subset in subsets:
            for neuron in subset:
                redundant_neurons_set.add(tuple(neuron))  # We use tuple because lists are not hashable

    # Calculate and print the percentage of redundant neurons
    percentage_redundant_neurons = len(redundant_neurons_set) / total_neurons * 100
    print(f"Percentage of redundant neurons: {percentage_redundant_neurons}%")

    # Prepare a dictionary to count neurons
    neuron_count = {}
    for neuron in redundant_neurons_set:
        if neuron not in neuron_count:
            neuron_count[neuron] = 0
        neuron_count[neuron] += 1

    # You can then use neuron_count to plot a histogram or other visualization.
import matplotlib.pyplot as plt
from collections import Counter

def plot_redundant_neurons_distribution(redundant_neurons):
    # Prepare a counter to count neurons by layer
    layer_counter = Counter()

    # Count all unique redundant neurons
    for _, subsets in redundant_neurons.items():
        for subset in subsets:
            for neuron in subset:
                layer_counter.update([neuron[0]])

    # Prepare data for the plot
    layers = sorted(layer_counter.keys())
    percentages = [layer_counter[layer] / sum(layer_counter.values()) * 100 for layer in layers]

    # Plot
    plt.figure(figsize=(8, 3))
    plt.bar(layers, percentages, width=1.02, color='#0165fc')
    plt.xlabel('Layer', fontsize=20)
    plt.ylabel('Percentage', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
