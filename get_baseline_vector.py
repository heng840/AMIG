import json

import numpy as np
import torch
import os
from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
from knowledge_neurons.patch import register_hook


class Baseline_specific_KnowledgeNeurons(KnowledgeNeurons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_specific_baseline_vector(self, language, layer_idx, output_path):
        baseline_sentences = {
            'en': 'The sky is yellow. Apples grow under the ground. Birds live in the sea.',
            'zh': '天空是黄色的。苹果在地下生长。鸟儿住在海里。'
        }
        base_sentence = baseline_sentences[language]
        encoded_input = self.tokenizer(base_sentence, return_tensors="pt").to(self.device)

        _, baseline_activations, = self.get_baseline_with_activations(encoded_input, layer_idx)
        baseline_vector = baseline_activations.detach().cpu().numpy().tolist()

        with open(output_path, 'w') as f:
            json.dump(baseline_vector, f)

    def get_baseline_with_activations(self, encoded_input: dict, layer_idx: int, mask_idx: int = None):
        def get_activations(model, layer_idx):
            def hook_fn(acts):
                self.baseline_activations = acts.mean(dim=1)

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations


class Baseline_average_KnowledgeNeurons(KnowledgeNeurons):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_avg_baseline_vector(self, dataset_path, output_path, layer_idx, step_size):
        # Load your dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Extract prompts and calculate embeddings
        prompts = []
        for key, value in dataset.items():
            for group in value:
                for triplet in group:
                    prompt = triplet[0]
                    prompts.append(prompt)

        # Generate embeddings
        embeddings = []
        for j in range(0, len(prompts), step_size):
            # Prepare inputs for the model
            encoded_input, mask_idx, _ = self._prepare_inputs(prompts[j])
            _, baseline_activations, = self.get_baseline_with_activations(encoded_input, layer_idx=layer_idx,
                                                                          mask_idx=mask_idx)
            embeddings.append(baseline_activations.detach().cpu().numpy().tolist())

        # Compute the average over all embeddings
        avg_baseline_vector = np.mean(embeddings, axis=0).tolist()

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(avg_baseline_vector, f)


if __name__ == "__main__":
    # Load BERT model and tokenizer
    # model_name = "bert-base-multilingual-cased"
    model_name = "ai-forever/mGPT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup model + tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_name)
    os.makedirs('datasets/baseline_vector/gpt/specific_en', exist_ok=True)
    os.makedirs('datasets/baseline_vector/gpt/specific_zh', exist_ok=True)
    os.makedirs('datasets/baseline_vector/gpt/average_en', exist_ok=True)
    os.makedirs('datasets/baseline_vector/gpt/average_zh', exist_ok=True)
    # baseline_kn_spe = Baseline_specific_KnowledgeNeurons(model, tokenizer, model_type=model_type(model_name))
    # for i in range(12):
    #     baseline_kn_spe.generate_specific_baseline_vector(
    #         language='en', layer_idx=i, output_path=f'datasets/baseline_vector/gpt/specific_en/layer{i}.json')
    #     baseline_kn_spe.generate_specific_baseline_vector(
    #         language='zh', layer_idx=i, output_path=f'datasets/baseline_vector/gpt/specific_zh/layer{i}.json')
    baseline_kn_ave = Baseline_average_KnowledgeNeurons(model, tokenizer, model_type=model_type(model_name))
    for i in range(baseline_kn_ave.n_layers()):
        baseline_kn_ave.generate_avg_baseline_vector(
            dataset_path='../my_m_kn/datasets/en/data_all_allbags.json',
            layer_idx=i,
            output_path=f'datasets/baseline_vector/gpt/average_en/layer{i}.json', step_size=200)
        baseline_kn_ave.generate_avg_baseline_vector(
            dataset_path='../my_m_kn/datasets/zh/data_all_allbags.json',
            layer_idx=i, output_path=f'datasets/baseline_vector/gpt/average_zh/layer{i}.json', step_size=200)
