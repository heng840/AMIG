import einops
import numpy as np
import torch
from torch import nn
from torch.distributions import Uniform
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
import copy
import torch.nn.functional as F
from knowledge_neurons import KnowledgeNeurons
from knowledge_neurons.patch import patch_ff_layer, unpatch_ff_layer

np.random.seed(42)


class garns(KnowledgeNeurons):
    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            model_type: str = "bert",
            device: str = None,
            # Any other parameters for GARNs
    ):
        super().__init__(model, tokenizer, model_type, device)

    "下面的注释部分的代码，是非线性插值的办法，没有什么作用。"

    # @staticmethod
    # def compute_alpha(diff_activations, p):
    #     """
    #     Compute the segment lengths and the cumulative alphas for each interpolated input.
    #     `diff_activations`: torch.Tensor
    #     `p`: int
    #     """
    #     segment_lengths = 1.0 / (torch.abs(diff_activations) ** p)
    #     total_length = torch.sum(segment_lengths)
    #     alpha = torch.cumsum(segment_lengths, dim=0) / total_length
    #
    #     return alpha

    # def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
    #                  layer_idx=0, encoded_input=None, mask_idx=None, p=1):
    #     """
    #     Compute the interpolated inputs along one randomly selected path.
    #     `p`: int
    #         The power to which the absolute difference is raised for computing segment lengths.
    #     """
    #     diff_activations = activations  # Assuming activations as x here
    #
    #     # Compute segment lengths and alpha
    #     segment_lengths = torch.pow(torch.abs(diff_activations), p)
    #     total_length = torch.sum(segment_lengths)
    #     alpha = torch.cumsum(segment_lengths, dim=0) / total_length
    #
    #     # Compute interpolated inputs
    #     interpolated_inputs = alpha * diff_activations
    #     # Here, x'_i is considered as 0
    #
    #     # Scale interpolated_inputs over steps
    #     out = interpolated_inputs * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
    #
    #     return out

    def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None):
        """
        SIG方法，最有效。
        """
        # emb: (1, ffn_size)
        if self.model_type == 'bert':
            replace_token_id = self.tokenizer.mask_token_id
        else:
            replace_token_id = self.tokenizer.eos_token_id

        all_res = []

        # get original activations for the complete input
        _, original_activations = self.get_baseline_with_activations(encoded_input,
                                                                     layer_idx=layer_idx, mask_idx=mask_idx)

        for idx in range(encoded_input['input_ids'].size(1)):
            # create a copy of the input and replace the idx-th word with mask token
            masked_input = copy.deepcopy(encoded_input)
            masked_input['input_ids'][0][idx] = replace_token_id

            # get masked activations to use as baseline
            _, baseline_activations = self.get_baseline_with_activations(masked_input,
                                                                         layer_idx=layer_idx, mask_idx=mask_idx)
            step = (original_activations - baseline_activations) / steps  # (1, ffn_size)

            res = torch.cat([torch.add(baseline_activations, step * i) for i in range(steps)], dim=0)
            all_res.append(res)
        # average
        mean_res = torch.stack(all_res).mean(dim=0)
        return mean_res

    # def distill_attributions(self, selected_neurons, attribution_scores):
    #     # Add the new filter rule here
    #     drop_neurons = [neuron for neuron in selected_neurons if neuron[0] < self.model.config.n_layer - 4]
    #     drop_neurons.sort(key=lambda x: attribution_scores[tuple(x)], reverse=True)
    #
    #     # Group neurons by layer
    #     grouped_neurons = {}
    #     for neuron in drop_neurons:
    #         layer = neuron[0]
    #         if layer not in grouped_neurons:
    #             grouped_neurons[layer] = []
    #         grouped_neurons[layer].append(neuron)
    #
    #     # Pick the first one from each group (highest score per layer)
    #     keep_neurons = [neurons[0] for neurons in grouped_neurons.values()]
    #
    #     # If there are more than 5 neurons, keep only the top 5
    #     if len(keep_neurons) > 5:
    #         keep_neurons = keep_neurons[:5]
    #
    #     # Keep neurons from the last few layers
    #     keep_layers_neurons = [neuron for neuron in selected_neurons if neuron[0] >= self.model.config.n_layer - 4]
    #
    #     # Final selection of neurons
    #     selected_neurons = keep_layers_neurons + keep_neurons
    #
    #     return selected_neurons

    def get_coarse_neurons(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            threshold: float = None,
            adaptive_threshold: float = None,
            percentile: float = None,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ):
        """
        """

        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            baseline_vector_path=baseline_vector_path,
            normalization=normalization
        )
        # sorted_attribution_scores, \
        #     sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)
        #
        # # Convert indices to original shape
        # original_shape_indices = np.unravel_index(sorted_attribution_indice.cpu().numpy(), attribution_scores.shape)

        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        threshold = attribution_scores.max().item() * adaptive_threshold

        selected_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
        # Sort neurons based on attribution scores
        selected_neurons.sort(key=lambda x: attribution_scores[x[0]][x[1]], reverse=True)
        return selected_neurons
        # if self.model_type == 'bert':
        #     return selected_neurons
        # else:
        #     selected_neurons = self.distill_attributions(selected_neurons=selected_neurons,
        #                                                  attribution_scores=attribution_scores)
        #     return selected_neurons

    def get_scores_for_layer(
            self,
            prompt,
            ground_truth,
            encoded_input=None,
            layer_idx=0,
            batch_size=10,
            steps=20,
            attribution_method="integrated_grads",
            baseline_vector_path=None,
            delta=10,
            epsilon=100,  # 不能过滤掉大的数值。
            k_path=1
    ):
        """
        """
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, ground_truth)
        n_sampling_steps = len(target_label)
        baseline_outputs, baseline_activations = self.get_baseline_with_activations(encoded_input, layer_idx, mask_idx)

        n_batches = steps // batch_size

        # Initialize an accumulator for the Distilled Gradients
        D_accumulator = torch.zeros_like(baseline_activations.squeeze(0))

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )

            if n_sampling_steps > 1:
                argmax_next_token = (
                    baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                )
                next_token_str = self.tokenizer.decode(argmax_next_token)
            scaled_weights = self.scaled_input(encoded_input=encoded_input, steps=steps, layer_idx=layer_idx,
                                               mask_idx=mask_idx)
            for k in range(k_path):  # Repeat K times for K paths
                # sorted_scaled_weights, \
                #     sorted_scaled_indice = torch.sort(scaled_weights.flatten(), descending=True)
                if k_path > 1:
                    noise = Uniform(torch.tensor([0.0]), torch.tensor([0.1])).sample(scaled_weights.shape).to(
                        self.device).squeeze(-1)
                     # the randomness is applied to each step,
                    # noise = Uniform(torch.tensor([0.0]), torch.tensor([0.1])).sample(baseline_activations.shape).to(self.device).squeeze(-1)
                    # , the randomness is applied to the entire sequence of steps.
                    scaled_weights = (1 - noise) * scaled_weights + noise * baseline_activations
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    target_idx = target_label[i]
                    grad = torch.autograd.grad(torch.unbind(probs[:, target_idx]), batch_weights)[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps

                if n_sampling_steps > 1:
                    prompt += next_token_str
                # sorted_scores, \
                #     sorted_indices = torch.sort(integrated_grads_this_step.flatten(), descending=True)
                D_accumulator += integrated_grads_this_step

                # sorted_d_A, \
                #     sorted_d_A_indices = torch.sort(D_accumulator.flatten(), descending=True)
        # Normalize by m*K to get the final attribution for each feature
        A_i = D_accumulator / k_path

        return A_i

    def get_refined_neurons(
            self,
            prompts,
            ground_truth,
            negative_examples=None,
            p=0.5,
            batch_size=10,
            steps=20,
            coarse_adaptive_threshold=0.3,
            coarse_threshold=None,
            coarse_percentile=None,
            quiet=False,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ):
        refined_neurons = []
        for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            refined_neurons.extend(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                    baseline_vector_path=baseline_vector_path,
                    normalization=normalization,
                    k_path=k_path
                )
            )
        return refined_neurons


# class threshold_and_drop_kn(KnowledgeNeurons):
#     def __init__(
#             self,
#             model: nn.Module,
#             tokenizer: PreTrainedTokenizerBase,
#             model_type: str = "bert",
#             device: str = None,
#             # Any other parameters for GARNs
#     ):
#         super().__init__(model, tokenizer, model_type, device)
#
#     def get_coarse_neurons(
#             self,
#             prompt: str,
#             ground_truth: str,
#             batch_size: int = 10,
#             steps: int = 20,
#             threshold: float = None,
#             adaptive_threshold: float = None,
#             percentile: float = None,
#             attribution_method: str = "integrated_grads",
#             pbar: bool = True,
#             baseline_vector_path=None,
#             normalization=False,
#     ):
#         attribution_scores = self.get_scores(
#             prompt,
#             ground_truth,
#             batch_size=batch_size,
#             steps=steps,
#             pbar=pbar,
#             attribution_method=attribution_method,
#             baseline_vector_path=baseline_vector_path,
#             normalization=normalization
#         )
#         sorted_attribution_scores, \
#             sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)
#         threshold = attribution_scores.max().item() * adaptive_threshold
#         selected_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
#
#         # Add the new filter rule here
#         drop_neurons = [neuron for neuron in selected_neurons if (neuron[0] == 0 or neuron[0] == 1)]
#         if drop_neurons:
#             drop_neurons.sort(key=lambda x: attribution_scores[tuple(x)], reverse=True)
#             selected_neurons = [drop_neurons[0]] + [neuron for neuron in selected_neurons if neuron[0] != 0]
#
#         return selected_neurons


# class threshold_and_drop_all_kn(KnowledgeNeurons):
#     def __init__(
#             self,
#             model: nn.Module,
#             tokenizer: PreTrainedTokenizerBase,
#             model_type: str = "bert",
#             device: str = None,
#             # Any other parameters for GARNs
#     ):
#         super().__init__(model, tokenizer, model_type, device)
#
#     def get_coarse_neurons(
#             self,
#             prompt: str,
#             ground_truth: str,
#             batch_size: int = 10,
#             steps: int = 20,
#             threshold: float = None,
#             adaptive_threshold: float = None,
#             percentile: float = None,
#             attribution_method: str = "integrated_grads",
#             pbar: bool = True,
#             baseline_vector_path=None,
#             normalization=False,
#     ):
#         attribution_scores = self.get_scores(
#             prompt,
#             ground_truth,
#             batch_size=batch_size,
#             steps=steps,
#             pbar=pbar,
#             attribution_method=attribution_method,
#             baseline_vector_path=baseline_vector_path,
#             normalization=normalization
#         )
#         sorted_attribution_scores, \
#             sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)
#         threshold = attribution_scores.max().item() * adaptive_threshold
#         selected_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
#
#         # Add the new filter rule here
#         drop_neurons = [neuron for neuron in selected_neurons if (neuron[0] == 0 or neuron[0] == 1)]
#         if drop_neurons:
#             drop_neurons.sort(key=lambda x: attribution_scores[tuple(x)], reverse=True)
#             selected_neurons = [neuron for neuron in selected_neurons if neuron[0] != 0]
#
#         return selected_neurons
#
#
# class threshold_and_sort_kn(KnowledgeNeurons):
#     def __init__(
#             self,
#             model: nn.Module,
#             tokenizer: PreTrainedTokenizerBase,
#             model_type: str = "bert",
#             device: str = None,
#             # Any other parameters for GARNs
#     ):
#         super().__init__(model, tokenizer, model_type, device)
#
#     def get_coarse_neurons(
#             self,
#             prompt: str,
#             ground_truth: str,
#             batch_size: int = 10,
#             steps: int = 20,
#             threshold: float = None,
#             adaptive_threshold: float = None,
#             percentile: float = None,
#             attribution_method: str = "integrated_grads",
#             pbar: bool = True,
#             baseline_vector_path=None,
#             normalization=False,
#             k_path=1
#     ):
#         adaptive_threshold = adaptive_threshold / 10
#         attribution_scores = self.get_scores(
#             prompt,
#             ground_truth,
#             batch_size=batch_size,
#             steps=steps,
#             pbar=pbar,
#             attribution_method=attribution_method,
#             baseline_vector_path=baseline_vector_path,
#             normalization=normalization,
#             k_path=k_path
#         )
#         sorted_attribution_scores, \
#             sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)
#         threshold = attribution_scores.max().item() * adaptive_threshold
#         selected_neurons = torch.nonzero(attribution_scores > threshold).cpu().tolist()
#         sorted_neurons = [[int(item) for item in items] for items in zip(*selected_neurons)]
#         return sorted_neurons
