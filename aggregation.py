"""
aggregation.py — Token aggregation strategy and feature extraction.

Strategy:
- 5 selected layers: [12, 21, 22, 23, 24]
- For each layer: mean pooling over ALL real tokens
- For each layer: mean pooling over LAST 30% tokens (approx response zone)
- Max pooling over final layer
- Last token of final layer (strong signal in decoder-only models)
- Feature dim: 5*896 (full mean) + 5*896 (response mean) + 896 (max) + 896 (last) = 10752
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

_SELECTED_LAYERS = [12, 21, 22, 23, 24]


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    mask_float = attention_mask.float().unsqueeze(-1)  # (seq_len, 1)
    n_real = mask_float.sum().clamp(min=1.0)

    # Response zone mask: last 30% of real tokens
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_response = max(1, int(n_real.item() * 0.30))
    response_positions = real_positions[-n_response:]
    response_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    response_mask[response_positions] = 1.0
    response_mask_f = response_mask.unsqueeze(-1)
    n_response_f = response_mask_f.sum().clamp(min=1.0)

    parts = []

    for layer_idx in _SELECTED_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)

        # Full mean pooling
        mean_vec = (layer * mask_float).sum(dim=0) / n_real
        parts.append(mean_vec)

        # Response-zone mean pooling
        resp_vec = (layer * response_mask_f).sum(dim=0) / n_response_f
        parts.append(resp_vec)

    # Max pooling on final layer
    last_layer = hidden_states[-1]
    neg_inf_mask = (attention_mask == 0).unsqueeze(-1).expand_as(last_layer)
    last_masked = last_layer.masked_fill(neg_inf_mask, float("-inf"))
    max_vec = last_masked.max(dim=0).values
    parts.append(max_vec)

    # Last real token of final layer (strong summary signal in decoder-only)
    last_pos = int(real_positions[-1].item())
    last_tok_vec = last_layer[last_pos]
    parts.append(last_tok_vec)

    return torch.cat(parts, dim=0)
    # dim = 5*2*896 + 896 + 896 = 9856 + 1792 = 10752... actually 5*2=10 vecs + 2 = 12*896=10752


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    attention_mask = attention_mask.to(device)

    mask_float = attention_mask.float().unsqueeze(-1)
    n_real = mask_float.sum().clamp(min=1.0)

    norms = []
    mean_vecs = []
    for layer_idx in _SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        mean_vec = (layer * mask_float).sum(dim=0) / n_real
        mean_vecs.append(mean_vec)
        norms.append(mean_vec.norm().unsqueeze(0))

    norms_tensor = torch.cat(norms)

    cos_sims = []
    for i in range(len(mean_vecs) - 1):
        sim = F.cosine_similarity(
            mean_vecs[i].unsqueeze(0),
            mean_vecs[i + 1].unsqueeze(0),
        )
        cos_sims.append(sim)
    cos_sims_tensor = torch.stack(cos_sims)

    seq_len = torch.tensor([torch.log(n_real + 1.0)], device=device)
    std_norms = torch.tensor([norms_tensor.std()], device=device)

    return torch.cat([norms_tensor, cos_sims_tensor, seq_len, std_norms], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
