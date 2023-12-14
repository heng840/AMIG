# launch with `python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE pararel_evaluate.py`
import glob

from knowledge_neurons import (
    initialize_model_and_tokenizer,
    model_type,
    pararel_expanded,
    ALL_MODELS,
    KnowledgeNeurons,
)
import random
from functools import lru_cache
from tqdm import tqdm
import json
import argparse
import torch
from pathlib import Path
import os

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
        # default='bert-base-cased',
        # default='gpt2',
        help=f"name of the LM to use - choose from {ALL_MODELS}",
    )

    parser.add_argument('--pararel_json', type=str,
                        default='datasets/correspond_dataset/en.json')
    parser.add_argument('--neurons_result_dir', type=str, default='7_18_add_n_sample/mbert/en_threshold_0.3_0.2to0.25')
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

    parser.add_argument(
        "--adaptive_threshold",
        type=float,
        default=0.3,
        help="A setting used to determine the score threshold above which coarse neurons are selected "
             "- the paper uses 0.3",
        # todo 修改评价指标，top k%和bottom k%。
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="the threshold for the sharing percentage - we retain neurons that are shared by p% of prompts "
             "(p here is a decimal fraction, i.e between 0 and 1)",
    )
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--debug_keys', type=list, default=['P39', 'P264', 'P37', 'P108', 'P131', 'P103']
                        , help='debug选取了哪几个relation name, []代表不进行debug')
    parser.add_argument('--baseline_vector_path', type=str,
                        default=None,
                        )
    parser.add_argument('--normalization', action='store_true', default=False)
    parser.add_argument('--k_percent', type=int, default=-1, help='-1 表示用阈值法')
    parser.add_argument('--bart', default='encoder', type=str, choices=['encoder', 'decoder'])
    # parser.add_argument('--garns_ablation', type=str, default='garns',
    #                     choices=['wo_random', 'garns', 'wo_distill', 'wo_N_U'])
    # parser.add_argument('--synergistic_threshold_percent_low', type=float, default=0.2)
    # parser.add_argument('--synergistic_threshold_percent_high', type=float, default=0.25)
    # parser.add_argument('--with_synergistic', action='store_true', default=True)  # fixme 注意修改这里
    parser.add_argument('--k_path', type=int, default=1)

    args = parser.parse_args()
    if args.baseline_vector_path == 'None':
        args.baseline_vector_path = None

    RESULTS_DIR = Path(args.neurons_result_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(args.seed)

    # load dataset
    # each item in pararel is the same 'fact' (head/relation/tail) expressed in different ways
    PARAREL = pararel_expanded(data_path=args.pararel_json)
    # if args.debug_keys:
    #     PARAREL = {key: PARAREL[key] for key in PARAREL if PARAREL[key].get("relation_name") in args.debug_keys}
    #     with open('datasets/source_data/zh_PARAREL_debug.json', "w", encoding='utf-8') as f:
    #         json.dump(PARAREL, f, ensure_ascii=False)
    # else:
    #     PARAREL = kn_bag_dict

    ##############################################################################
    # data parallel stuff
    NUM_REPLICAS = torch.cuda.device_count()
    INDICES = list(range(len(PARAREL)))
    INDICES = INDICES[args.local_rank: len(PARAREL): NUM_REPLICAS]
    KEYS = list(PARAREL.keys())
    torch.cuda.set_device(args.local_rank)
    ##############################################################################

    # initialize results dicts
    RESULTS = {}
    NEURONS = {}
    # setup model + tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    # if args.garns_ablation == 'wo_random':
    #     from knowledge_neurons.ablation.garns_wo_random import garns
    # elif args.garns_ablation == 'wo_distill':
    #     from knowledge_neurons.ablation.garns_wo_distill import garns
    # elif args.garns_ablation == 'wo_N_U':
    #     from knowledge_neurons.ablation.garns_wo_N_U_interpolation import garns
    # else:
    # from knowledge_neurons.garns import garns
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(args.model_name))


    # because we may end up getting some neurons multiple times, use lru cache to save time
    @lru_cache(maxsize=None)
    def get_neurons(_uuid):
        PROMPTS, GROUND_TRUTH, RELATION_NAME = (
            PARAREL[_uuid]["sentences"],
            PARAREL[_uuid]["obj_label"],
            PARAREL[_uuid]["relation_name"],
        )
        neurons = kn.get_refined_neurons(
            prompts=PROMPTS,
            ground_truth=GROUND_TRUTH.lower(),
            p=args.p,
            batch_size=args.batch_size,
            steps=args.steps,
            coarse_adaptive_threshold=args.adaptive_threshold,
            quiet=True,
            baseline_vector_path=args.baseline_vector_path,
            normalization=args.normalization,
            k_path=args.k_path,
        )
        return neurons, PARAREL[_uuid]


    def get_sorted_neurons(_uuid):
        PROMPTS, GROUND_TRUTH, RELATION_NAME = (
            PARAREL[_uuid]["sentences"],
            PARAREL[_uuid]["obj_label"],
            PARAREL[_uuid]["relation_name"],
        )
        neurons = kn.get_sorted_neurons(
            prompts=PROMPTS,
            ground_truth=GROUND_TRUTH.lower(),
            batch_size=args.batch_size,
            steps=args.steps,
            quiet=True,
            baseline_vector_path=args.baseline_vector_path,
            normalization=False
        )
        # k_neurons = neurons[int(len(neurons) * args.k_percent / 100):]  # bottom
        k_neurons = neurons[:int(len(neurons) * args.k_percent / 10000)]  # top
        refined_k_neurons = k_neurons
        # refined_k_neurons = kn.get_sorted_refined_neurons(all_sorted_neurons=k_neurons, p=args.p)
        return refined_k_neurons, PARAREL[_uuid]


    def get_unrelated_fact(KEYS, uuid):
        n_keys = len(KEYS)
        while True:
            random_uuid = KEYS[random.randint(0, n_keys - 1)]
            if random_uuid == uuid:
                continue
            return random_uuid


    results_folder = args.neurons_result_dir
    processed_uuids = set()
    # Get a list of all result files
    result_files = glob.glob(os.path.join(results_folder, '*results*.json'))
    # For each results file...
    for results_file in result_files:
        # Open the results file
        with open(results_file, 'r') as f:
            # Load the results as a dict
            results = json.load(f)
        # Add the UUIDs from this file to the set of processed UUIDs
        processed_uuids.update(results.keys())
    # go through each item in the PARAREL dataset, get the refined neurons, save them,
    # and evaluate the results when suppressing the
    # refined neurons vs. unrelated neurons.
    for i, idx in enumerate(tqdm(INDICES, position=args.local_rank)):
        uuid = KEYS[idx]
        if uuid in processed_uuids:
            continue
        try:
            unrelated_uuid = get_unrelated_fact(
                KEYS, uuid
            )  # get a uuid for an unrelated fact / relation

            if args.k_percent == -1:
                neurons, data = get_neurons(uuid)  # get refined neurons
                unrelated_neurons, unrelated_data = get_neurons(
                    unrelated_uuid
                )  # get the unrelated neurons
            else:  # sorted + refine
                neurons, data = get_sorted_neurons(uuid)  # get refined neurons
                unrelated_neurons, unrelated_data = get_sorted_neurons(unrelated_uuid)
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory! in{uuid}")
            continue
        # initialize a results dict
        results_this_uuid = {
            "suppression": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    # "log_odds": [],  # and this
                    # "comp_score": [],  # added
                    # "suff_score": [],  # added
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    # "log_odds": [],  # and this
                    # "comp_score": [],  # added
                    # "suff_score": [],  # added
                    "n_prompts": len(unrelated_data["sentences"]),
                }},
            "enhancement": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    # "log_odds": [],  # and this
                    # "comp_score": [],  # added
                    # "suff_score": [],  # added
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    # "log_odds": [],  # and this
                    # "comp_score": [],  # added
                    # "suff_score": [],  # added
                    "n_prompts": len(unrelated_data["sentences"]),
                }}
        }
        synergistic_neurons_this_uuid = []

        temp_record = {}
        for PROMPT in data["sentences"]:
            gt = data["obj_label"].lower()
            # if args.with_synergistic:
            # synergistic_neurons = kn.test_synergistic_neurons(
            #     PROMPT, gt.lower(), neurons[: 20], threshold_percent_low=args.synergistic_threshold_percent_low,
            #     threshold_percent_high=args.synergistic_threshold_percent_high)
            # temp_record[uuid] = synergistic_neurons
            # with open('test_synergistic_results.json', 'w') as f:
            #     json.dump(temp_record, f)
            # synergistic_neurons_this_uuid.extend(synergistic_neurons)
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

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["related"]["correct_before"].append(
                suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["related"]["correct_after"].append(
                suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["related"]["correct_before"].append(
                enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["related"]["correct_after"].append(
                enhancement_results["after"]["argmax_completion"] == gt
            )
        for PROMPT in unrelated_data["sentences"]:
            # do the same but with unrelated facts

            gt = unrelated_data["obj_label"].lower()

            unrelated_suppression_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )
            unrelated_enhancement_results, _ = kn.enhance_knowledge(
                PROMPT, gt, neurons, quiet=True
            )

            # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
            suppression_prob_diff = (unrelated_suppression_results["after"]["gt_prob"] -
                                     unrelated_suppression_results["before"]["gt_prob"]) / \
                                    unrelated_suppression_results["before"]["gt_prob"]
            results_this_uuid["suppression"]["unrelated"]["pct_change"].append(suppression_prob_diff)
            enhancement_prob_diff = (unrelated_enhancement_results["after"]["gt_prob"] -
                                     unrelated_enhancement_results["before"]["gt_prob"]) / \
                                    unrelated_enhancement_results["before"]["gt_prob"]
            results_this_uuid["enhancement"]["unrelated"]["pct_change"].append(enhancement_prob_diff)

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["unrelated"]["correct_before"].append(
                unrelated_suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["unrelated"]["correct_after"].append(
                unrelated_suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["unrelated"]["correct_before"].append(
                unrelated_enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["unrelated"]["correct_after"].append(
                unrelated_enhancement_results["after"]["argmax_completion"] == gt
            )

        results_this_uuid["n_refined_neurons"] = len(neurons)
        results_this_uuid["n_unrelated_neurons"] = len(unrelated_neurons)
        results_this_uuid["relation_name"] = data["relation_name"]
        results_this_uuid['neurons'] = neurons
        # if args.with_synergistic:
        RESULTS[uuid] = {
            **results_this_uuid,
            'synergistic_neurons_this_uuid': synergistic_neurons_this_uuid
        }
        # else:
        #     RESULTS[uuid] = {**results_this_uuid}

        NEURONS[uuid] = {'neurons': neurons,
                         'relation_name': data["relation_name"]}

        neurons_json_path = RESULTS_DIR / f"temp_neurons_{args.local_rank}.json"
        results_json_path = RESULTS_DIR / f"temp_results_{args.local_rank}.json"
        with open(neurons_json_path, "a") as f:
            json.dump(NEURONS, f, indent=4)
        with open(results_json_path, "a") as f:
            json.dump(RESULTS, f, indent=4)
    neurons_json_path = RESULTS_DIR / f"pararel_neurons_{args.local_rank}.json"
    results_json_path = RESULTS_DIR / f"pararel_results_{args.local_rank}.json"
    redundant_neurons_path = RESULTS_DIR / f'redundant_neurons_{args.local_rank}.json'
    with open(neurons_json_path, "w") as f:
        json.dump(NEURONS, f, indent=4)
    with open(results_json_path, "w") as f:
        json.dump(RESULTS, f, indent=4)
