# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Union, List
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from knowledge_neurons import (
    initialize_model_and_tokenizer,
    model_type,
    pararel_expanded,
    ALL_MODELS,
    garns,
)


def aggregate_results(data_dir):
    aggregated_results = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename)) as file:
            data = json.load(file)
            for key in data:
                for relation, value in data[key].items():
                    if relation != 'P364':
                        aggregated_results[key][relation] += value
    return aggregated_results

def load_json_files_from_directory(dir_path, keyword):
    result = {}
    for filename in os.listdir(dir_path):
        if keyword in filename:
            with open(os.path.join(dir_path, filename)) as file:
                data = json.load(file)
                result.update(data)
    return result


def main(args):
    if args.baseline_vector_path == 'None':
        args.baseline_vector_path = None
    RESULTS_DIR = Path(args.with_synergistic_neurons_result_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)
    with open(args.wrong_fact_dataset_json) as file:
        dataset = json.load(file)
    torch.cuda.set_device(args.local_rank)
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    kn = garns(model, tokenizer, model_type=model_type(args.model_name))
    NUM_REPLICAS = torch.cuda.device_count()
    ALL_UUIDS = list(dataset.keys())
    INDICES = list(range(len(ALL_UUIDS)))
    INDICES = INDICES[args.local_rank: len(ALL_UUIDS): NUM_REPLICAS]
    # Load the wrong fact dataset and synergistic neurons
    synergistic_neurons_result = load_json_files_from_directory(args.with_synergistic_neurons_result_dir,
                                                                'pararel_results')

    # Group data by relation_name
    data_grouped_by_relation = defaultdict(list)
    for uuid in dataset:
        item = dataset[uuid]
        relation_name = item['relation_name']
        data_grouped_by_relation[relation_name].append((uuid, item))

    # Separate synergistic_neurons_result by relation_name
    synergistic_neurons_result_grouped = defaultdict(dict)
    for uuid, result in synergistic_neurons_result.items():
        item = dataset[uuid]
        relation_name = item['relation_name']
        synergistic_neurons_result_grouped[relation_name][uuid] = result

    # Split data and count neurons for each relation_name
    filtered_neurons_by_relation = defaultdict(set)
    for relation_name, items in data_grouped_by_relation.items():
        train_items, _ = train_test_split(items, test_size=0.2, random_state=42)

        # For training items, count the neurons and filter out those exceeding the threshold
        neuron_counts = defaultdict(int)
        for uuid, item in train_items:
            if uuid not in synergistic_neurons_result_grouped[relation_name]:
                continue
            synergistic_neurons = synergistic_neurons_result_grouped[relation_name][uuid][
                "synergistic_neurons_this_uuid"]

            # Flatten the list of neuron pairs to count individual neurons
            for neuron in [tuple(neuron_pair) for pair in synergistic_neurons for neuron_pair in pair]:
                neuron_counts[neuron] += 1

        # Filter out neurons based on threshold
        threshold_count = int(max(neuron_counts.values()) * args.threshold_filter_DN)
        filtered_neurons = {neuron for neuron, count in neuron_counts.items() if count > threshold_count}
        filtered_neurons_by_relation[relation_name] = filtered_neurons

    # Initialize counters
    def load_existing_results(data_dir):
        existing_results = aggregate_results(data_dir)
        return existing_results

    # Load existing results from the directory
    existing_results_1 = load_existing_results(data_dir='temp1')  # Change 'temp3' to the correct directory path
    existing_results_2 = load_existing_results(data_dir='temp2')  # Change 'temp3' to the correct directory path

    # Initialize your result variables with the existing results
    correct_true_by_relation = defaultdict(int, existing_results_1["correct_true_by_relation"])
    correct_true_by_relation_directly_use_PLMs = defaultdict(int, existing_results_2["correct_true_by_relation"])
    total_true_by_relation = defaultdict(int, existing_results_1["total_true_by_relation"])
    correct_false_by_relation = defaultdict(int, existing_results_1["correct_false_by_relation"])
    correct_false_by_relation_directly_use_PLMs = defaultdict(int, existing_results_2["correct_false_by_relation"])
    total_false_by_relation = defaultdict(int, existing_results_1["total_false_by_relation"])
    # correct_true_by_relation = defaultdict(int)
    # total_true_by_relation = defaultdict(int)
    # correct_false_by_relation = defaultdict(int)
    # total_false_by_relation = defaultdict(int)
    # correct_true_by_relation_directly_use_PLMs = defaultdict(int)
    # correct_false_by_relation_directly_use_PLMs = defaultdict(int)
    # with open('hallucination_temp.json', 'r') as f:
    #     saved_results = json.load(f)
    # Iterate over test set for each relation_name

    # aggregate_results_1 = aggregate_results('temp1')
    # aggregate_results_2 = aggregate_results('temp2')
    for relation_name, items in data_grouped_by_relation.items():
        if relation_name in existing_results_1['correct_true_by_relation']:
            print(f"Using existing results for relation {relation_name}. Continuing...")
            continue
        _, test_items = train_test_split(items, test_size=0.2, random_state=42)
        test_indices = list(range(len(test_items)))
        test_indices = test_indices[args.local_rank: len(test_items): NUM_REPLICAS]
        all_uuids_processed_successfully = True
        # Iterate over each item in the test set
        for i, test_idx in enumerate(tqdm(test_indices, position=args.local_rank)):
            uuid, item = test_items[test_idx]
        # for uuid, item in tqdm(test_items):
            # Replace synergistic_neurons with the filtered neurons for this relation
            synergistic_neurons = filtered_neurons_by_relation[relation_name]
            if not synergistic_neurons or uuid == 'c58aea67-b0e4-423b-86ad-d9acd169ab16':
                continue
            try:
                # Perform fact-checking with the filtered neurons
                correct_true, correct_false = kn.test_detection_system(
                    item=item, synergistic_neurons=synergistic_neurons,
                    threshold=args.threshold,
                    batch_size=args.batch_size, steps=args.steps,
                    baseline_vector_path=args.baseline_vector_path,
                    score_path=args.score_path
                )
                correct_true_by_relation[relation_name] += correct_true
                total_true_by_relation[relation_name] += 1
                correct_false_by_relation[relation_name] += correct_false
                total_false_by_relation[relation_name] += 1

                # Perform fact-checking directly using PLMs
                correct_true_directly_use_PLMs, correct_false_directly_use_PLMs = kn.test_detection_system_directly_use_PLMs(
                    item=item)
                correct_true_by_relation_directly_use_PLMs[relation_name] += correct_true_directly_use_PLMs
                correct_false_by_relation_directly_use_PLMs[relation_name] += correct_false_directly_use_PLMs
            except torch.cuda.OutOfMemoryError:
                print(f"Skipping {uuid} due to CUDA out of memory error.")
                all_uuids_processed_successfully = False
                continue
        if all_uuids_processed_successfully:
            temp_result_dict = {
                "correct_true_by_relation": dict(correct_true_by_relation),
                "total_true_by_relation": dict(total_true_by_relation),
                "correct_false_by_relation": dict(correct_false_by_relation),
                "total_false_by_relation": dict(total_false_by_relation),
            }
            os.makedirs('temp3', exist_ok=True)
            os.makedirs('temp4', exist_ok=True)
            with open(f'temp3/hallucination_temp_{args.local_rank}.json', 'w') as f:
                json.dump(temp_result_dict, f)
            temp_result_dict_directly_use_PLMs = {
                "correct_true_by_relation": dict(correct_true_by_relation_directly_use_PLMs),
                "total_true_by_relation": dict(total_true_by_relation),
                "correct_false_by_relation": dict(correct_false_by_relation_directly_use_PLMs),
                "total_false_by_relation": dict(total_false_by_relation),
            }
            with open(f'temp4/hallucination_temp_directly_use_PLMs_{args.local_rank}.json', 'w') as f:
                json.dump(temp_result_dict_directly_use_PLMs, f)

    # Save the results
    result_dict = {
        "correct_true_by_relation": dict(correct_true_by_relation),
        "total_true_by_relation": dict(total_true_by_relation),
        "correct_false_by_relation": dict(correct_false_by_relation),
        "total_false_by_relation": dict(total_false_by_relation),
    }
    result_dict_directly_use_PLMs = {
        "correct_true_by_relation": dict(correct_true_by_relation_directly_use_PLMs),
        "total_true_by_relation": dict(total_true_by_relation),
        "correct_false_by_relation": dict(correct_false_by_relation_directly_use_PLMs),
        "total_false_by_relation": dict(total_false_by_relation),
    }
    with open(f'{args.with_synergistic_neurons_result_dir}/hallucination_{args.local_rank}.json', 'w') as f:
        json.dump(result_dict, f)
    with open(f'{args.with_synergistic_neurons_result_dir}/hallucination_directly_use_PLMs_{args.local_rank}.json',
              'w') as f:
        json.dump(result_dict_directly_use_PLMs, f)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--local-rank", help="local rank for multigpu processing", type=int, default=0
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="facebook/mbart-large-50",
        # default='facebook/bart-large',
        # default='bert-base-multilingual-cased',
        default="ai-forever/mGPT",
        # default='bert-base-uncased',
        # default='gpt2',
        help=f"name of the LM to use - choose from {ALL_MODELS}",
    )

    parser.add_argument('--wrong_fact_dataset_json', type=str,
                        # default='datasets/correspond_dataset/wrong_fact_en.json',
                        default='datasets/correspond_dataset/wrong_fact_zh.json',
                        )
    parser.add_argument('--with_synergistic_neurons_result_dir', type=str,
                        default='723final/mbert/zh_threshold_0.2_0.2to0.25',
                        # default='723final/mbert/en_threshold_0.3_0.2to0.25',
                        )
    parser.add_argument("--batch_size", type=int, default=20,
                        help='# assert steps % batch_size == 0'
                             '# n_batches = steps // batch_size，'
                             '此外，对应gpt类，因为要re-tokenize new inputs，所以需要内存更多')
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of steps to run the integrated gradient calculation for",
    )

    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--baseline_vector_path', type=str,
                        default=None,
                        )
    parser.add_argument('--score_path', type=str,
                        default='threshold_tune/score_mgpt_zh.json',
                        )
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--threshold_filter_DN', type=float,
                        default=0.7,
                        )

    args = parser.parse_args()
    # os.makedirs('723/gpt_0.3_0.2to0.25/hallucination_acc', exist_ok=True)
    # calculate_total_accuracy(result_dir='723/gpt_0.3_0.2to0.25t/hallucination_acc',
    #                          json_files="723/gpt_0.3_0.2to0.25/en_threshold_0.3_0.2to0.25/hallucination_src0.3.json")
    main(args=args)

