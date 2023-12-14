# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`
import argparse
import json
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from knowledge_neurons import (
    initialize_model_and_tokenizer,
    model_type,
    ALL_MODELS,
    garns,
)


def merge_datasets(en_json, zh_json):
    # Load the English dataset
    with open(en_json, 'r') as f:
        en_data = json.load(f)

    # Load the Chinese dataset
    with open(zh_json, 'r') as f:
        zh_data = json.load(f)

    # Initialize the multilingual dataset
    multilingual_data = {}

    # For each entry in the English dataset
    for _uuid in en_data:
        # If the same _uuid exists in the Chinese dataset
        if _uuid in zh_data:
            # Merge the English and Chinese data for this _uuid
            multilingual_data[_uuid] = {
                'english': en_data[_uuid],
                'chinese': zh_data[_uuid]
            }

    return multilingual_data


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
        default='bert-base-multilingual-cased',
        # default="ai-forever/mGPT",
        # default='bert-base-uncased',
        # default='gpt2',
        help=f"name of the LM to use - choose from {ALL_MODELS}",
    )

    parser.add_argument('--en_json', type=str,
                        default='datasets/correspond_dataset/en.json')
    parser.add_argument('--zh_json', type=str,
                        default='datasets/correspond_dataset/zh.json')
    parser.add_argument('--en_neurons', type=str,
                        default='7_13_with_synergistic/mbert/en_threshold_0.3_0.2to0.25',
                        )
    parser.add_argument('--zh_neurons', type=str,
                        default='7_13_with_synergistic/mbert/zh_threshold_0.2_0.2to0.25'
                        )
    parser.add_argument('--result_dir', type=str,
                        # default='Debug_res/test_mbert/edit_zh2en'
                        default='7_13_with_synergistic/mbert/cross_language_results_0.2_0.3',
                        )
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    args = parser.parse_args()
    en_neurons = {}
    zh_neurons = {}

    for file in os.listdir(args.en_neurons):
        if 'neurons' in file:
            with open(os.path.join(args.en_neurons, file), 'r') as f:
                en_neurons.update(json.load(f))
    for file in os.listdir(args.zh_neurons):
        if 'neurons' in file:
            with open(os.path.join(args.zh_neurons, file), 'r') as f:
                zh_neurons.update(json.load(f))
    RESULTS_DIR = Path(args.result_dir)
    random.seed(args.seed)
    # Load the neuron_data

    MULTILINGUAL_PARAREL = merge_datasets(args.en_json, args.zh_json)
    KEYS = list(MULTILINGUAL_PARAREL.keys())

    NUM_REPLICAS = torch.cuda.device_count()
    INDICES = list(range(len(MULTILINGUAL_PARAREL)))
    INDICES = INDICES[args.local_rank: len(MULTILINGUAL_PARAREL): NUM_REPLICAS]
    torch.cuda.set_device(args.local_rank)
    ##############################################################################

    # initialize results dicts
    RESULTS = {}
    # setup model + tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    kn = garns(model, tokenizer, model_type=model_type(args.model_name))


    def get_neurons_from_file(monolingual_neurons, _uuid):
        if _uuid not in monolingual_neurons:
            return None, None
        neurons = monolingual_neurons[_uuid]['neurons']
        return neurons, MULTILINGUAL_PARAREL[_uuid]



    def compare_edit(edit_type):
        # go through each item in the PARAREL dataset, get the refined neurons, save them,
        # and evaluate the results when suppressing the
        # refined neurons vs. unrelated neurons.
        if edit_type == 'en2zh':
            monolingual_neurons = en_neurons
        else:
            monolingual_neurons = zh_neurons
        for i, idx in enumerate(tqdm(INDICES, position=args.local_rank)):
            uuid = KEYS[idx]

            def get_unrelated_fact(KEYS, uuid):
                n_keys = len(KEYS)
                while True:
                    random_uuid = KEYS[random.randint(0, n_keys - 1)]
                    if random_uuid != uuid and random_uuid in monolingual_neurons:
                        return random_uuid

            unrelated_uuid = get_unrelated_fact(
                KEYS, uuid
            )  # get a uuid for an unrelated fact / relation
            if uuid not in monolingual_neurons:
                continue
            neurons, data = get_neurons_from_file(_uuid=uuid, monolingual_neurons=monolingual_neurons)
            if neurons is None:
                continue
            _, unrelated_data = get_neurons_from_file(
                _uuid=unrelated_uuid, monolingual_neurons=monolingual_neurons
            )

            # initialize a results dict
            results_this_uuid = {
                "suppression": {
                    "related": {
                        "pct_change": [],
                    },
                    "unrelated": {
                        "pct_change": [],
                    }},
                "enhancement": {
                    "related": {
                        "pct_change": [],
                    },
                    "unrelated": {
                        "pct_change": [],
                    }}
            }
            if edit_type == 'en2zh':  # 编辑英文神经元，因此对比中文的prompt。
                language = 'chinese'
            else:
                language = 'english'

            for PROMPT in data[language]["sentences"]:
                gt = data[language]["obj_label"].lower()
                # really should be using a different for the suppression, but the authors didn't make their bing dataset available
                suppression_results, _ = kn.suppress_knowledge(PROMPT, gt, neurons, quiet=True)
                enhancement_results, _ = kn.enhance_knowledge(PROMPT, gt, neurons, quiet=True)

                # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
                suppression_prob_diff = (suppression_results["after"]["gt_prob"] - suppression_results["before"][
                    "gt_prob"]) / suppression_results["before"]["gt_prob"]
                results_this_uuid["suppression"]["related"]["pct_change"].append(suppression_prob_diff)

                enhancement_prob_diff = (enhancement_results["after"]["gt_prob"] - enhancement_results["before"][
                    "gt_prob"]) / enhancement_results["before"]["gt_prob"]
                results_this_uuid["enhancement"]["related"]["pct_change"].append(enhancement_prob_diff)
            for PROMPT in unrelated_data[language]["sentences"]:
                gt = unrelated_data[language]["obj_label"].lower()
                # really should be using a different for the suppression, but the authors didn't make their bing dataset available
                suppression_results, _ = kn.suppress_knowledge(PROMPT, gt, neurons, quiet=True)
                enhancement_results, _ = kn.enhance_knowledge(PROMPT, gt, neurons, quiet=True)

                # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
                suppression_prob_diff = (suppression_results["after"]["gt_prob"] - suppression_results["before"][
                    "gt_prob"]) / suppression_results["before"]["gt_prob"]
                results_this_uuid["suppression"]["unrelated"]["pct_change"].append(suppression_prob_diff)

                enhancement_prob_diff = (enhancement_results["after"]["gt_prob"] - enhancement_results["before"][
                    "gt_prob"]) / enhancement_results["before"]["gt_prob"]
                results_this_uuid["enhancement"]["unrelated"]["pct_change"].append(enhancement_prob_diff)

            # results_this_uuid["n_refined_neurons"] = len(neurons)
            # results_this_uuid["n_unrelated_neurons"] = len(unrelated_neurons)
            assert data['english']["relation_name"] == data['chinese']["relation_name"]
            results_this_uuid["relation_name"] = data['english']["relation_name"]
            RESULTS[uuid] = {**results_this_uuid}
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_json_path = RESULTS_DIR / f"{edit_type}_{args.local_rank}.json"
        with open(results_json_path, "w") as f:
            json.dump(RESULTS, f, indent=4)

    # compare_edit(edit_type='en2zh')
    compare_edit(edit_type='zh2en')
