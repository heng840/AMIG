import einops
import numpy as np
import torch
from torch import nn
from torch.distributions import Uniform
from transformers import PreTrainedTokenizerBase

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

    def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None, p=1):
        """
        Compute the interpolated inputs along one randomly selected path.
        `p`: int
            The power to which the absolute difference is raised for computing segment lengths.
        """

        out = activations * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]

        return out

    @staticmethod
    def distill_attributions(integrated_grads, delta_percent, epsilon_percent):
        """
        Compute the Distilled Gradients and aggregate the attributions for a single path.
        Use percentiles of the integrated gradients as adaptive thresholds.
        """
        sorted_scaled_integrated_grads, \
            sorted_scaled_integrated_grads_indice = torch.sort(integrated_grads.flatten(), descending=True)
        # Compute the adaptive thresholds
        delta = np.percentile(integrated_grads.cpu().detach().numpy(), delta_percent)
        epsilon = np.percentile(integrated_grads.cpu().detach().numpy(), epsilon_percent)

        # Compute the mask
        mask = (integrated_grads >= delta) & (integrated_grads <= epsilon)
        # mask = integrated_grads >= delta
        # Apply the mask to IG to get the distilled gradients
        D_ik = integrated_grads * mask

        sorted_scaled_D_ik, \
            sorted_scaled_D_ik_indice = torch.sort(D_ik.flatten(), descending=True)

        return D_ik

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
            m=10,
            K=10
    ):
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, ground_truth, )
        baseline_outputs, baseline_activations = self.get_baseline_with_activations(encoded_input, layer_idx, mask_idx)

        n_batches = steps // batch_size

        integrated_grads = []

        # Initialize an accumulator for the Distilled Gradients
        D_accumulator = torch.zeros_like(baseline_activations.squeeze(0))
        for k in range(K):  # Repeat K times for K paths
            # TODO 加入随机路径。现在K条path完全一致，这样循环没有意义。

            scaled_weights = self.scaled_input(activations=baseline_activations, steps=steps,
                                               device=self.device,
                                               baseline_vector_path=baseline_vector_path,
                                               layer_idx=layer_idx)

            noise = Uniform(torch.tensor([0.0]), torch.tensor([0.1])).sample(scaled_weights.shape).to(self.device).squeeze(-1)
            sorted_scaled_weights, \
                sorted_scaled_indice = torch.sort(scaled_weights.flatten(), descending=True)

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
                target_idx = target_label
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

            sorted_scores, \
                sorted_indices = torch.sort(integrated_grads_this_step.flatten(), descending=True)
            # Compute the distilled gradients for this path
            D_ik = self.distill_attributions(integrated_grads=integrated_grads_this_step,
                                             delta_percent=delta, epsilon_percent=epsilon)

            sorted_d, \
                sorted_d_indices = torch.sort(D_ik.flatten(), descending=True)
            D_accumulator += D_ik

            sorted_d_A, \
                sorted_d_A_indices = torch.sort(D_accumulator.flatten(), descending=True)
        # Normalize by m*K to get the final attribution for each feature
        A_i = D_accumulator / (m * K)

        return A_i


