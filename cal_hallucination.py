import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd


def aggregate_results(data_dir):
    aggregated_results = defaultdict(lambda: defaultdict(int))
    aggregated_results_use_PLMs = defaultdict(lambda: defaultdict(int))
    for filename in os.listdir(data_dir):
        if 'hallucination_directly_use_PLMs_' in filename:
            with open(os.path.join(data_dir, filename)) as file:
                data = json.load(file)
                for key in data:
                    if key in ["correct_true_by_relation", "total_true_by_relation",
                               "correct_false_by_relation", "total_false_by_relation"]:
                        for relation, value in data[key].items():
                            aggregated_results_use_PLMs[key][relation] += value
        if 'hallucination' in filename and 'hallucination_directly_use_PLMs_' not in filename:
            with open(os.path.join(data_dir, filename)) as file:
                data = json.load(file)
                for key in data:
                    if key in ["correct_true_by_relation", "total_true_by_relation",
                               "correct_false_by_relation", "total_false_by_relation"]:
                        for relation, value in data[key].items():
                            aggregated_results[key][relation] += value
    return aggregated_results, aggregated_results_use_PLMs

def load_json_files_from_directory(dir_path, keyword):
    result = {}
    for filename in os.listdir(dir_path):
        if keyword in filename:
            with open(os.path.join(dir_path, filename)) as file:
                data = json.load(file)
                result.update(data)
    return result


def compute_metrics(results):
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)

    metrics = []
    for relation in results["correct_true_by_relation"].keys():
        true_positives = results["correct_true_by_relation"][relation]
        true_negatives = results["correct_false_by_relation"][relation]
        total_positives = results["total_true_by_relation"][relation]
        total_negatives = results["total_false_by_relation"][relation]

        false_positives = total_positives - true_positives
        false_negatives = total_negatives - true_negatives

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else np.nan
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else np.nan
        f1_score = 2 * ((precision * recall) / (precision + recall)) if precision + recall > 0 else np.nan
        accuracy = (true_positives + true_negatives) / (total_positives + total_negatives)

        metrics.append({
            "Relation": relation,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "Accuracy": accuracy
        })
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


# Function to compute overall metrics
def compute_overall_metrics(results):
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)

    total_true_positives = sum(results["correct_true_by_relation"].values())
    total_true_negatives = sum(results["correct_false_by_relation"].values())
    total_positives = sum(results["total_true_by_relation"].values())
    total_negatives = sum(results["total_false_by_relation"].values())

    total_false_positives = total_positives - total_true_positives
    total_false_negatives = total_negatives - total_true_negatives

    overall_precision = total_true_positives / (total_true_positives + total_false_positives) if total_true_positives + total_false_positives > 0 else np.nan
    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if total_true_positives + total_false_negatives > 0 else np.nan
    overall_f1_score = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)) if overall_precision + overall_recall > 0 else np.nan
    overall_accuracy = (total_true_positives + total_true_negatives) / (total_positives + total_negatives)

    overall_metrics = pd.DataFrame([{
        "Overall Precision": overall_precision,
        "Overall Recall": overall_recall,
        "Overall F1 Score": overall_f1_score,
        # "Overall Accuracy": overall_accuracy
    }])
    return overall_metrics



if __name__ == "__main__":
    # hallucination_dir = '723final'
    hallucination_dir = '723'
    cases = [
        # 'mgpt/zh',
        # 'mbert/en_threshold_0.3_0.2to0.25',
        # 'mbert/zh_threshold_0.2_0.2to0.25',
        # 'mgpt/en_threshold_0.3_0.2to0.25',
        # 'mgpt/zh_threshold_0.2_0.2to0.25',
        'bert_0.3_0.2to0.25',
        'gpt_0.3_0.2to0.25',
    ]
    # Initialize empty DataFrames
    metrics_df = pd.DataFrame()
    overall_metrics_df = pd.DataFrame()

    # Process results and save to CSV
    for case in cases:
        folder = os.path.join(hallucination_dir, case)
        existing_results, existing_results_use_PLMs = aggregate_results(folder)

        # Compute metrics and append to DataFrame
        folder_metrics_df = compute_metrics(existing_results)
        folder_metrics_df_use_PLMs = compute_metrics(existing_results_use_PLMs)
        folder_metrics_df['Folder'] = folder  # add a column to indicate the folder
        folder_metrics_df_use_PLMs['Folder'] = f'{folder}_use_PLMs'
        metrics_df = metrics_df._append(folder_metrics_df, ignore_index=True)
        metrics_df = metrics_df._append(folder_metrics_df_use_PLMs, ignore_index=True)

        # Compute overall metrics and append to DataFrame
        folder_overall_metrics_df = compute_overall_metrics(existing_results)
        folder_overall_metrics_df_use_PLMs = compute_overall_metrics(existing_results_use_PLMs)
        folder_overall_metrics_df['Folder'] = folder  # add a column to indicate the folder
        folder_overall_metrics_df_use_PLMs['Folder'] = f'{folder}_use_PLMs'
        overall_metrics_df = overall_metrics_df._append(folder_overall_metrics_df, ignore_index=True)
        overall_metrics_df = overall_metrics_df._append(folder_overall_metrics_df_use_PLMs, ignore_index=True)

    # Save the final DataFrames to CSV files
    metrics_df.to_csv(f'{hallucination_dir}/combined_metrics.csv', index=False)
    # overall_metrics_df.to_csv(f'{hallucination_dir}/combined_overall_metrics.csv', index=False)

    # main_experiment_metrics = compute_metrics(f'{hallucination_dir}/hallucination_0.json')
    # comparative_experiment_metrics = compute_metrics(f'{hallucination_dir}/hallucination_directly_use_PLMs_0.json')
    # compute_overall_metrics(f'{hallucination_dir}/hallucination_0.json')
    # compute_overall_metrics(f'{hallucination_dir}/hallucination_directly_use_PLMs_0.json')
    # data = prepare_data_for_plotting(main_experiment_metrics, comparative_experiment_metrics)
    # os.makedirs(f'{hallucination_dir}/figs', exist_ok=True)
    # plot_data(data, out_path=f'{hallucination_dir}/figs/hallucination.png')  fixme 统一在plot_sub_res里
