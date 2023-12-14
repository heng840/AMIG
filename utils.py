import json

import torch


def normalize_attributions(attributions):
    min_val = torch.min(attributions)
    max_val = torch.max(attributions)
    normalized_attributions = (attributions - min_val) / (max_val - min_val)
    return normalized_attributions


def get_baseline_vector(baseline_vector_path):
    with open(baseline_vector_path, 'r') as f:
        baseline_vector = json.load(f)
    return baseline_vector


def generate_avg_baseline_vector(dataset_path, output_path, max_seq_length):
    """
    base_sentence = 'The sky is [MASK].'
    这样使得mask_idx有用，
    The mask_idx variable stands for the index position of the masked token in the input sequence.

In transformer-based models like BERT, training is done using a technique known as "masked language modeling".
During this process, a certain percentage of the input tokens are masked out and the model is trained to predict the
original tokens from the context provided by the other, non-masked, tokens.

When you are feeding an input sequence to the model, you often want to pay attention to the output corresponding to a
specific token - often this token is the one that was masked out. This is where mask_idx comes into play.
 The variable mask_idx is used to index into the output tensor and extract the embeddings (activations) of the masked token.

In the function get_baseline_with_activations, the mask_idx is used in the hook function to extract the intermediate
activations of the masked token at a given layer in the transformer model.
This allows the function to return the baseline output and activations of the model at the specified layer and for the masked token.
    """
    pass
