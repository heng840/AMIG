import torch


def eval_log_odds(
     model,
     input_ids,
     attention_mask,
     neuron_attributions,
     original_prediction,
     perturb_func,
     top_k,
):
    # Get original logits
    original_logits = model(input_ids, attention_mask=attention_mask)[0]
    original_prob = torch.softmax(original_logits, dim=-1)

    # Make a copy of the input IDs for perturbation
    perturbed_input_ids = input_ids.clone().detach()

    # Get the indices of the top-k neurons
    topk_indices = torch.topk(neuron_attributions, int(neuron_attributions.shape[0] * top_k / 100), sorted=False).indices

    # Apply the perturbation function to the selected neurons
    perturbed_input_ids[0, topk_indices] = perturb_func(perturbed_input_ids[0, topk_indices])

    # Get the logits for the perturbed input
    perturbed_logits = model(perturbed_input_ids, attention_mask=attention_mask)[0]
    perturbed_prob = torch.softmax(perturbed_logits, dim=-1)

    # Calculate the log-odds score
    log_odds_score = (torch.log(perturbed_prob[0, original_prediction]) - torch.log(original_prob[0, original_prediction])).item()

    return log_odds_score



def eval_sufficiency(
    forward_fn,
    input_embed,
    position_embed,
    type_embed,
    attention_mask,
    attr,
    topk=20,
):
    logits_original = forward_fn(
        input_embed,
        attention_mask=attention_mask,
        position_embed=position_embed,
        type_embed=type_embed,
        return_all_logits=True,
    ).squeeze()

    predicted_label = torch.argmax(logits_original).item()
    prob_original = torch.softmax(logits_original, dim=0)

    topk_indices = torch.topk(
        attr, int(attr.shape[0] * topk / 100), sorted=False
    ).indices
    if len(topk_indices) == 0:
        # topk% is too less to select even word - so no masking will happen.
        return 0

    mask = torch.zeros_like(input_embed[0][:, 0]).bool()
    mask[topk_indices] = 1
    masked_input_embed = input_embed[0][mask].unsqueeze(0)
    masked_attention_mask = (
        None
        if attention_mask is None
        else attention_mask[0][mask].unsqueeze(0)
    )
    masked_position_embed = (
        None
        if position_embed is None
        else position_embed[0][: mask.sum().item()].unsqueeze(0)
    )
    masked_type_embed = (
        None if type_embed is None else type_embed[0][mask].unsqueeze(0)
    )

    logits_perturbed = forward_fn(
        masked_input_embed,
        attention_mask=masked_attention_mask,
        position_embed=masked_position_embed,
        type_embed=masked_type_embed,
        return_all_logits=True,
    ).squeeze()

    prob_perturbed = torch.softmax(logits_perturbed, dim=0)

    return (
        prob_original[predicted_label] - prob_perturbed[predicted_label]
    ).item()


def eval_comprehensiveness(
    forward_fn,
    input_embed,
    position_embed,
    type_embed,
    attention_mask,
    attr,
    topk=20,
):
    logits_original = forward_fn(
        input_embed,
        attention_mask=attention_mask,
        position_embed=position_embed,
        type_embed=type_embed,
        return_all_logits=True,
    ).squeeze()

    predicted_label = torch.argmax(logits_original).item()
    prob_original = torch.softmax(logits_original, dim=0)

    topk_indices = torch.topk(
        attr, int(attr.shape[0] * topk / 100), sorted=False
    ).indices

    mask = torch.ones_like(input_embed[0][:, 0]).bool()
    mask[topk_indices] = 0
    masked_input_embed = input_embed[0][mask].unsqueeze(0)
    masked_attention_mask = (
        None
        if attention_mask is None
        else attention_mask[0][mask].unsqueeze(0)
    )
    masked_position_embed = (
        None
        if position_embed is None
        else position_embed[0][: mask.sum().item()].unsqueeze(0)
    )
    masked_type_embed = (
        None if type_embed is None else type_embed[0][mask].unsqueeze(0)
    )

    logits_perturbed = forward_fn(
        masked_input_embed,
        attention_mask=masked_attention_mask,
        position_embed=masked_position_embed,
        type_embed=masked_type_embed,
        return_all_logits=True,
    ).squeeze()

    prob_perturbed = torch.softmax(logits_perturbed, dim=0)

    return (
        prob_original[predicted_label] - prob_perturbed[predicted_label]
    ).item()
