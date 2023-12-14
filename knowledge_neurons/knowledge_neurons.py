import itertools
import math
import os
import random
from functools import partial
from typing import Optional, Tuple

import einops
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, RobertaForSequenceClassification

from .patch import *
import json


class KnowledgeNeurons:
    def __init__(
            self,
            model: nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            model_type: str = "bert",
            device: str = None,
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer

        self.baseline_activations = None
        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)
        elif "gpt" in self.model_type:
            self.model_type = 'gpt'
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wpe"
            # add pad token
            # new_tokens = ['<PAD>']
            # self.tokenizer.add_tokens(new_tokens, special_tokens=True)
            # self.model.resize_token_embeddings(len(tokenizer))
            # self.tokenizer.pad_token = '<PAD>'
        elif self.model_type == 'bart_encoder':
            self.transformer_layers_attr = "model.encoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2.weight"
            self.word_embeddings_attr = "model.encoder.embed_tokens.weight"

        elif self.model_type == 'bart_decoder':
            self.transformer_layers_attr = "model.decoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2.weight"
            self.word_embeddings_attr = "model.decoder.embed_tokens.weight"


        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, ):
        # tokenized_sequence = self.tokenizer.tokenize(prompt)
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "gpt" in self.model_type or 'bart_decoder' in self.model_type:
            # mask_idx = encoded_input["input_ids"].shape[-1] - 1
            # Remove [MASK] token from prompt for GPT models
            prompt = prompt.replace("[MASK] .", "").strip()
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # Here, the model is expected to predict the next word(s) after the prompt
            mask_idx = - 1
        else:
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
            # mask_indices = torch.where(
            #     encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            # )
        if target is not None:
            # y = self.tokenizer.encode(target)
            # if "gpt" in self.model_type:
            target_label = self.tokenizer.encode(target)
            # else:
            #     target_label = self.tokenizer.convert_tokens_to_ids(target)
        else:
            target_label = None
        return encoded_input, mask_idx, target_label

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        n_sampling_steps = len(target_label)
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label[i]
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            x = self.model.config.intermediate_size
            return self.model.config.intermediate_size
        else:
            y = self.model.config.hidden_size
            return self.model.config.hidden_size * 4

    def scaled_input(self, activations=None, steps=20, device="cpu", baseline_vector_path=None,
                     layer_idx=0, encoded_input=None, mask_idx=None):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """

        def get_baseline_vector(baseline_vector_path):
            with open(baseline_vector_path, 'r') as f:
                baseline_vector = json.load(f)
            return baseline_vector

        if baseline_vector_path is None:
            diff_activations = activations
        else:
            base_vector = torch.Tensor(get_baseline_vector(f'{baseline_vector_path}/layer{layer_idx}.json')).to(
                device)
            diff_activations = activations - base_vector
        tiled_activations = einops.repeat(diff_activations, "b d -> (r b) d", r=steps)

        out = (tiled_activations * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None])
        return out

    def calc_attributions(self, prompt: str, ground_truth: str, scaled_inputs: torch.Tensor,
                          diff_activations: torch.Tensor, quiet: bool = False):
        # for deeplift
        self.model.eval()
        self.model.zero_grad()
        steps = scaled_inputs.shape[0]

        # Run the model on scaled_inputs and compute the output and gradients
        _, all_hidden_states, _, _ = self._generate_with_custom_inputs(
            scaled_inputs.view(steps, -1, self.model.config.hidden_size), ground_truth)
        hidden_states_grad = torch.autograd.grad(all_hidden_states[-1].sum(), scaled_inputs, retain_graph=True)[0]

        # Compute the DeepLift attributions
        attributions = (diff_activations * hidden_states_grad).mean(0)

        return attributions

    def get_baseline_with_activations(
            self, encoded_input, layer_idx, mask_idx
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                # for i in range(12):
                #     x= acts[:,i,:]
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations  # todo 查看维度，确定SIG如何加入。

    def get_scores(
            self,
            prompt: str,
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            attribution_method: str = "integrated_grads",
            pbar: bool = True,
            baseline_vector_path=None,
            normalization=False,
            k_path=1,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        scores = []
        # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        x = self.n_layers()
        for layer_idx in range(self.n_layers()):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
                baseline_vector_path=baseline_vector_path,
                k_path=k_path,
            )
            scores.append(layer_scores)
        stacked_scores = torch.stack(scores)
        return stacked_scores

    def get_sorted_neurons_one_prompt(self,
                                      prompt: str,
                                      ground_truth: str,
                                      batch_size: int = 10,
                                      steps: int = 20,
                                      adaptive_threshold=0.1,  # 二次过滤。 -1表示不过滤
                                      percentile: float = None,
                                      attribution_method: str = "integrated_grads",
                                      pbar: bool = True,
                                      baseline_vector_path=None,
                                      normalization=False
                                      ):
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            baseline_vector_path=baseline_vector_path,
            normalization=normalization,
        )
        flattened_scores = attribution_scores.flatten().detach().cpu().numpy()
        # Getting the sorted indices
        sorted_indices \
            = np.unravel_index(np.argsort(flattened_scores, axis=None)[::-1], attribution_scores.shape)
        # Creating list of lists [[layer, neuron], ...]
        sorted_neurons = [[int(item) for item in items] for items in zip(*sorted_indices)]
        return sorted_neurons

    def get_sorted_neurons(
            self,
            prompts: List[str],
            ground_truth: str,
            batch_size: int = 10,
            steps: int = 20,
            quiet=False,
            baseline_vector_path=None,
            normalization=False
    ):
        all_sorted_neurons = []
        for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            all_sorted_neurons.extend(
                self.get_sorted_neurons_one_prompt(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    pbar=False,
                    baseline_vector_path=baseline_vector_path,
                    normalization=normalization,
                )
            )

        return all_sorted_neurons

    @staticmethod
    def get_sorted_refined_neurons(all_sorted_neurons, len_prompts=1, p=0.3):
        """
        len_prompts在现在是1.后续拓展到单语义、多表达时，为n。
        """
        neuron_counts = collections.Counter()

        # count the frequency of each neuron
        for neuron in all_sorted_neurons:
            neuron_counts[tuple(neuron)] += 1

        refined_neurons = []
        for neuron, count in neuron_counts.items():
            if count > len_prompts * p:
                refined_neurons.append(list(neuron))

        return refined_neurons

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
            k_path=1,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold`
            (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            baseline_vector_path=baseline_vector_path,
            normalization=normalization,
            k_path=k_path
        )
        sorted_attribution_scores, \
            sorted_attribution_indice = torch.sort(attribution_scores.flatten(), descending=True)

        # Convert indices to original shape
        original_shape_indices = np.unravel_index(sorted_attribution_indice.cpu().numpy(), attribution_scores.shape)

        assert (
                sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        if threshold is not None:
            x = torch.nonzero(attribution_scores > threshold).cpu().tolist()
            len_x = len(x)
            return torch.nonzero(attribution_scores > threshold).cpu().tolist()
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            return (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist()
            )

    def get_refined_neurons(
            self,
            prompts: List[str],
            ground_truth: str,
            negative_examples: Optional[List[str]] = None,
            p: float = 0.5,
            batch_size: int = 10,
            steps: int = 20,
            coarse_adaptive_threshold: Optional[float] = 0.3,
            coarse_threshold: Optional[float] = None,
            coarse_percentile: Optional[float] = None,
            quiet=False,
            baseline_vector_path=None,
            normalization=False,
            k_path=1
    ) -> List[List[int]]:
        """
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
                prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
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
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                    negative_examples,
                    desc="Getting coarse neurons for negative examples",
                    disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                        baseline_vector_path=baseline_vector_path,
                        normalization=normalization
                    )
                )
        # 下面是二次过滤的代码
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
            self,
            prompt: str,
            ground_truth: str,
            layer_idx: int,
            batch_size: int = 10,
            steps: int = 20,
            attribution_method: str = "integrated_grads",
            baseline_vector_path=None,
            k_path=1
    ):
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth,
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = len(target_label)

        integrated_grads = []

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            (
                baseline_outputs,
                baseline_activations,
            ) = self.get_baseline_with_activations(
                encoded_input, layer_idx, mask_idx
            )
            if n_sampling_steps > 1:
                argmax_next_token = (
                    baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                )
                next_token_str = self.tokenizer.decode(argmax_next_token)

            scaled_weights = self.scaled_input(activations=baseline_activations, steps=steps,
                                               device=self.device,
                                               baseline_vector_path=baseline_vector_path,
                                               layer_idx=layer_idx, )
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
                if self.model_type == "bert":
                    inputs["token_type_ids"] = einops.repeat(
                        encoded_input["token_type_ids"],
                        "b d -> (r b) d",
                        r=batch_size,
                    )

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
                grad = torch.autograd.grad(
                    torch.unbind(probs[:, target_idx]), batch_weights
                )[0]
                grad = grad.sum(dim=0)
                integrated_grads_this_step.append(grad)

                unpatch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

            # then sum, and multiply by W-hat / m
            integrated_grads_this_step = torch.stack(
                integrated_grads_this_step, dim=0
            ).sum(dim=0)
            integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
            integrated_grads.append(integrated_grads_this_step)

            if n_sampling_steps > 1:
                prompt += next_token_str
        integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
            integrated_grads
        )
        return integrated_grads

    def modify_activations(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            mode: str = "suppress",
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the ground-truth being generated +
        # the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion:"
                f" `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch, so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the ground_truth being generated +
        # the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: "
                f"`{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def prediction_accuracy(
            self,
            prompt: str,
            ground_truth: str,
            neurons,
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this

        # patch model to suppress neurons
        # store all the layers we patch, so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode='suppress',
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )

        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def test_synergistic_neurons(self, prompt, ground_truth, all_neurons, threshold_percent_low,
                                 threshold_percent_high):
        """
        单独抑制：降低程度小于threshold_percent_low，协调抑制：下降程度大于threshold_percent_high
        """
        synergistic_neurons = []
        test_results = []
        # First pass: identify neurons whose removal doesn't impact accuracy
        potential_neurons = []
        original_results, _ = self.prediction_accuracy(prompt, ground_truth, [])
        original_accuracy = original_results["after"]["gt_prob"]

        for neuron in all_neurons:
            neurons_to_suppress = [neuron]
            new_results, _ = self.prediction_accuracy(prompt, ground_truth, neurons_to_suppress, undo_modification=True)
            new_accuracy = new_results["after"]["gt_prob"]
            test_results.append({'new_accuracy': new_accuracy,
                                 'original_accuracy * (1 - threshold_percent_low)': original_accuracy * (
                                         1 - threshold_percent_low),
                                 'synergistic_neurons_1': new_accuracy >= original_accuracy * (
                                         1 - threshold_percent_low)
                                 })

            if new_accuracy >= original_accuracy * (1 - threshold_percent_low):
                potential_neurons.append(neuron)

        # Second pass: identify pairs of potential neurons which, when removed together, do impact accuracy
        for pair in itertools.combinations(potential_neurons, 2):
            neurons_to_suppress = list(pair)
            new_results, _ = self.prediction_accuracy(prompt, ground_truth, neurons_to_suppress, undo_modification=True)
            new_accuracy_2 = new_results["after"]["gt_prob"]
            test_results.append({'new_accuracy_2': new_accuracy_2,
                                 'original_accuracy * (1 - threshold_percent_high)': original_accuracy * (
                                         1 - threshold_percent_high),
                                 'synergistic_neurons_2': new_accuracy_2 < original_accuracy * (
                                         1 - threshold_percent_high)
                                 })

            if new_accuracy_2 < original_accuracy * (1 - threshold_percent_high):
                synergistic_neurons.append(pair)
        # write results to json file
        return synergistic_neurons

    def test_redundant_neurons_all_todo(self, prompt, ground_truth, all_neurons, threshold):
        redundant_neurons = {}
        all_subsets = []
        for subset_size in range(1, len(all_neurons) + 1):
            for subset in itertools.combinations(all_neurons, subset_size):
                all_subsets.append(list(subset))
        # 原来的代码，会产生2^len(all_neurons)个子集。过于巨大了。但是这是下一步工作可以改进的地方，比如使用贪心算法等。 todo

        original_results, _ = self.prediction_accuracy(prompt, ground_truth, [])
        original_accuracy = original_results["after"]["gt_prob"]

        for subset in all_subsets:
            neurons_to_suppress = [neuron for neuron in all_neurons if neuron not in subset]
            new_results, _ = self.prediction_accuracy(prompt, ground_truth, neurons_to_suppress, undo_modification=True)
            new_accuracy = new_results["after"]["gt_prob"]
            if new_accuracy >= original_accuracy - threshold:
                if len(subset) in redundant_neurons:
                    redundant_neurons[len(subset)].append(subset)
                else:
                    redundant_neurons[len(subset)] = [subset]

        return redundant_neurons

    # def calc_log_odds(self, prompt: str, ground_truth: str, neurons: List[List[int]]):
    #     # Get the model's prediction on the original input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_original = outputs.logits
    #     prob_original = torch.softmax(logits_original, dim=-1)
    #     y_hat = self.tokenizer.encode(ground_truth, add_special_tokens=False)[0]
    #
    #     # Suppress top k% neurons
    #     _, unpatch_fn = self.modify_activations(
    #         prompt,
    #         ground_truth,
    #         neurons=neurons,
    #         mode="suppress",
    #         undo_modification=False,
    #         quiet=True,
    #     )
    #
    #     # Get the model's prediction on the modified input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_perturbed = outputs.logits
    #     prob_perturbed = torch.softmax(logits_perturbed, dim=-1)
    #
    #     # Calculate the log odds
    #     if self.model_type == "gpt" or self.model_type == "bart_decoder" or self.model_type == "bart_encoder":
    #         log_odds = torch.log(prob_perturbed[0, -1, y_hat]) - torch.log(prob_original[0, -1, y_hat])
    #     elif self.model_type == "bert":
    #         mask_idx = encoded_input['input_ids'].tolist()[0].index(self.tokenizer.mask_token_id)
    #         log_odds = torch.log(prob_perturbed[0, mask_idx, y_hat]) - torch.log(prob_original[0, mask_idx, y_hat])
    #     else:
    #         raise ValueError(
    #             f"Unexpected model_type: {self.model_type}. Expected 'gpt', 'bert', 'bart_encoder' or 'bart_decoder'.")
    #
    #     # Revert the modifications made to the model
    #     unpatch_fn()
    #
    #     return log_odds.item()
    #
    # def calc_comprehensiveness(self, prompt: str, ground_truth: str, neurons: List[List[int]]):
    #     # Get the model's prediction on the original input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_original = outputs.logits
    #     prob_original = torch.softmax(logits_original, dim=-1)
    #     y_hat = self.tokenizer.encode(ground_truth, add_special_tokens=False)[0]
    #
    #     # Suppress top k% neurons
    #     _, unpatch_fn = self.modify_activations(
    #         prompt,
    #         ground_truth,
    #         neurons=neurons,
    #         mode="suppress",
    #         undo_modification=False,
    #         quiet=True,
    #     )
    #
    #     # Get the model's prediction on the modified input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_perturbed = outputs.logits
    #     prob_perturbed = torch.softmax(logits_perturbed, dim=-1)
    #
    #     # Calculate the comprehensiveness score
    #     if self.model_type == "gpt" or self.model_type == "bart_decoder" or self.model_type == "bart_encoder":
    #         comp_score = prob_perturbed[0, -1, y_hat] - prob_original[0, -1, y_hat]
    #     elif self.model_type == "bert":
    #         mask_idx = encoded_input['input_ids'].tolist()[0].index(self.tokenizer.mask_token_id)
    #         comp_score = prob_perturbed[0, mask_idx, y_hat] - prob_original[0, mask_idx, y_hat]
    #     else:
    #         raise ValueError(f"Unexpected model_type: {self.model_type}.")
    #
    #     # Revert the modifications made to the model
    #     unpatch_fn()
    #
    #     return comp_score.item()
    #
    # def calc_sufficiency(self, prompt: str, ground_truth: str, neurons: List[List[int]], ):
    #     # Get the model's prediction on the original input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_original = outputs.logits
    #     prob_original = torch.softmax(logits_original, dim=-1)
    #     y_hat = self.tokenizer.encode(ground_truth, add_special_tokens=False)[0]
    #
    #     # Suppress all but top k% neurons
    #     _, unpatch_fn = self.modify_activations(
    #         prompt,
    #         ground_truth,
    #         neurons=neurons,
    #         mode="suppress",
    #         undo_modification=False,
    #         quiet=True,
    #     )
    #
    #     # Get the model's prediction on the modified input
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device=self.device)
    #     outputs = self.model(**encoded_input)
    #     logits_perturbed = outputs.logits
    #     prob_perturbed = torch.softmax(logits_perturbed, dim=-1)
    #
    #     # Calculate the sufficiency score
    #     if self.model_type == "gpt" or self.model_type == "bart_decoder" or self.model_type == "bart_encoder":
    #         suff_score = prob_perturbed[0, -1, y_hat] - prob_original[0, -1, y_hat]
    #     elif self.model_type == "bert":
    #         mask_idx = encoded_input['input_ids'].tolist()[0].index(self.tokenizer.mask_token_id)
    #         suff_score = prob_perturbed[0, mask_idx, y_hat] - prob_original[0, mask_idx, y_hat]
    #     else:
    #         raise ValueError(f"Unexpected model_type: {self.model_type}.")
    #
    #     # Revert the modifications made to the model
    #     unpatch_fn()
    #
    #     return suff_score.item()

    def suppress_knowledge(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting effect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
            self,
            prompt: str,
            ground_truth: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
            self,
            prompt: str,
            neurons: List[List[int]],
            target: str,
            mode: str = "edit",
            erase_value: str = "zero",
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            assert (
                    self.model_type == "bert"
            ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0]
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt":
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                # for BERT, BART encoder, and BART decoder, handle the weights this way.
                # BART follows the standard Transformer model, which uses Linear layers instead of Conv1D in its feed-forward blocks.
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt":
                    output_ff_weights[position, :] -= original_prediction_embedding * 2
                    output_ff_weights[position, :] += target_embedding * 2
                else:
                    output_ff_weights[:, position] -= original_prediction_embedding * 2
                    output_ff_weights[:, position] += target_embedding * 2
            else:
                if self.model_type == "gpt":
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt":
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    # for BERT, BART encoder, and BART decoder, handle the weights this way.
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
            self,
            prompt: str,
            target: str,
            neurons: List[List[int]],
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
            self,
            prompt: str,
            neurons: List[List[int]],
            erase_value: str = "zero",
            target: Optional[str] = None,
            undo_modification: bool = True,
            quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def hallucination_detection(self, prompt, label,
                                batch_size,
                                steps,
                                baseline_vector_path,
                                synergistic_neurons,
                                threshold,
                                relation_name,
                                score_path, ):
        """
        """
        # activations = self.get_baseline_with_activations(encoded_input=encoded_input, )
        scores = self.get_scores(prompt=prompt, ground_truth=label, batch_size=batch_size,
                                 steps=steps, baseline_vector_path=baseline_vector_path, )
        # Get the activations for the Synergistic Neurons
        # fixme
        # synergistic_neurons_tensor = torch.tensor(synergistic_neurons, dtype=torch.long)

        sn_scores = []
        for neuron in synergistic_neurons:
            score = scores[neuron[0], neuron[1]]
            sn_scores.append(score)

        sn_scores = torch.tensor(sn_scores)  # Convert list to tensor
        # Use tensor indexing to get the scores of synergistic neurons
        # sn_scores = scores[synergistic_neurons_tensor[:, :, 0], synergistic_neurons_tensor[:, :, 1]]

        x = sn_scores.mean().item()
        y = sn_scores.max().item()
        z = sn_scores.min().item()
        score_data = {
            "relation_name": relation_name,
            "mean_score": x,
            "max_score": y,
            "min_score": z
        }

        # Write score_data to a JSON file
        os.makedirs('threshold_tune', exist_ok=True)
        # Append score_data to a file
        with open(score_path, 'a') as f:
            f.write(json.dumps(score_data) + '\n')

        if sn_scores.mean().item() < threshold:
            return True
        else:
            return False

    def test_detection_system(self, item, synergistic_neurons, threshold,
                              batch_size, steps, baseline_vector_path, score_path):
        """
        Function to test the hallucination detection system on a single prompt.

        Parameters:
        - item: An item from the dataset, which includes a correct fact and a list of incorrect facts.
        - synergistic_neurons: The identified Synergistic Neurons for the item.
        - threshold: The activation threshold for the Synergistic Neurons to flag a potential hallucination.
        - batch_size: Batch size for gradient calculation.
        - steps: Steps for gradient calculation.
        - baseline_vector_path: Path to the baseline vector.

        Returns:
        - The accuracy of the detection system for true facts, false facts, and overall.
        """
        # Test the correct fact
        prompt = item['sentences'][0]
        true_label = item['obj_label']
        relation_name = item['relation_name']
        if not self.hallucination_detection(prompt=prompt, label=true_label,
                                            synergistic_neurons=synergistic_neurons,
                                            threshold=threshold, batch_size=batch_size, steps=steps,
                                            baseline_vector_path=baseline_vector_path,
                                            relation_name=f'True_{relation_name}',
                                            score_path=score_path):
            correct_true = 1
        else:

            correct_true = 0

        # Randomly select and test one incorrect fact
        random.seed(42)
        wrong_label = random.choice(item['wrong_fact'])
        if self.hallucination_detection(prompt=prompt, label=wrong_label, synergistic_neurons=synergistic_neurons,
                                        threshold=threshold, batch_size=batch_size, steps=steps,
                                        baseline_vector_path=baseline_vector_path,
                                        relation_name=f'False_{relation_name}',
                                        score_path=score_path):
            correct_false = 1
        else:
            correct_false = 0
        return correct_true, correct_false

    def hallucination_detection_directly_use_PLMs(self, prompt, label):
        """
        Function to detect potential hallucinations based on the prediction of the model.

        Parameters:
        - prompt: The input prompt for the model.
        - label: The expected correct output.

        Returns:
        - A boolean indicating whether the model's output is correct (True) or not (False).
        """
        # Check the model type
        if self.model_type == "gpt":
            # For GPT models, use the generate method
            prompt = prompt.replace("[MASK] .", "").strip()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            predicted_output_ids = self.model.generate(input_ids)
            predicted_output = self.tokenizer.decode(predicted_output_ids[0])
            predicted_output = predicted_output.split()[-1]
        elif self.model_type == "bert":
            # For BERT models, use the forward method
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]

            outputs = self.model(input_ids)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, mask_token_index, :]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]

            predicted_output = predicted_token
        else:
            raise ValueError("Unsupported model type")

        # Compare the predicted output with the correct label
        if predicted_output == label:
            return True
        else:
            return False

    def test_detection_system_directly_use_PLMs(self, item):
        """
        Function to test the hallucination detection system on a single prompt.

        Parameters:
        - item: An item from the dataset, which includes a correct fact and a list of incorrect facts.

        Returns:
        - The accuracy of the detection system for true facts, false facts, and overall.
        """
        # Test the correct fact
        prompt = item['sentences'][0]
        true_label = item['obj_label']
        if self.hallucination_detection_directly_use_PLMs(prompt=prompt, label=true_label):
            correct_true = 1
        else:
            correct_true = 0

        # Randomly select and test one incorrect fact
        random.seed(42)
        wrong_label = random.choice(item['wrong_fact'])
        if not self.hallucination_detection_directly_use_PLMs(prompt=prompt, label=wrong_label):
            correct_false = 1
        else:
            correct_false = 0
        return correct_true, correct_false

    @staticmethod
    def fact_check(claim, evidence):
        """
        Function to fact-check a claim based on the provided evidence.

        Parameters:
        - claim: The claim to verify.
        - evidence: The evidence to verify the claim.

        Returns:
        - A boolean indicating whether the evidence supports the claim (True) or not (False).
        """
        # Tokenize the claim with evidence

        # Load the tokenizer and model for fact-checking
        fact_check_tokenizer = RobertaTokenizer.from_pretrained('Dzeniks/roberta-fact-check')
        fact_check_model = RobertaForSequenceClassification.from_pretrained('Dzeniks/roberta-fact-check')

        inputs = fact_check_tokenizer.encode_plus(claim, evidence, return_tensors="pt")

        fact_check_model.eval()
        with torch.no_grad():
            prediction = fact_check_model(**inputs)

        label = torch.argmax(prediction[0]).item()

        # Return whether the evidence supports the claim
        if label == 0:
            return True
        else:
            return False

    def test_detection_system_use_fact_check(self, item):
        """
        Function to test the hallucination detection system on a single prompt.

        Parameters:
        - item: An item from the dataset, which includes a correct fact and a list of incorrect facts.

        Returns:
        - The accuracy of the detection system for true facts, false facts, and overall.
        """
        # Test the correct fact
        prompt = item['sentences'][0]
        true_label = item['obj_label']
        if self.fact_check(prompt, true_label):
            correct_true = 1
        else:
            correct_true = 0

        # Randomly select and test one incorrect fact
        random.seed(42)
        wrong_label = random.choice(item['wrong_fact'])
        if not self.fact_check(prompt, wrong_label):
            correct_false = 1
        else:
            correct_false = 0
        return correct_true, correct_false
