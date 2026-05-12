"""
Strategy:
- Select multiple informative layers (last 8 transformer layers + middle layers)
- Mean pooling over real tokens (non-padding) for each selected layer
- Max pooling over real tokens for the final layer
- Concatenate per-layer vectors into one feature vector
- Geometric features: per-layer activation norms, inter-layer cosine
  similarity (representation drift), and sequence length
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
_SELECTED_LAYERS = [8, 12, 16, 18, 20, 21, 22, 23, 24]
def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.
    """
    # Convert attention mask to float and unsqueeze to (seq_len, 1)
    mask_float = attention_mask.float().unsqueeze(-1)  # (seq_len, 1)
    n_real = mask_float.sum().clamp(min=1.0)            # scalar

    parts = []

    for layer_idx in _SELECTED_LAYERS:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)

        # Mean pooling over real tokens
        mean_vec = (layer * mask_float).sum(dim=0) / n_real  # (hidden_dim,)
        parts.append(mean_vec)

    # Max pooling over real tokens on the final layer
    last_layer = hidden_states[-1]  # (seq_len, hidden_dim)
    # Replace padding positions with -inf so they don't win the max
    neg_inf_mask = (attention_mask == 0).unsqueeze(-1).expand_as(last_layer)
    last_masked = last_layer.masked_fill(neg_inf_mask, float("-inf"))
    max_vec = last_masked.max(dim=0).values  # (hidden_dim,)
    parts.append(max_vec)

    return torch.cat(parts, dim=0)  # (len(_SELECTED_LAYERS)+1) * hidden_dim


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract hand-crafted geometric / statistical features from hidden states.
    """
    # Convert attention mask to float and unsqueeze to (seq_len, 1)
    mask_float = attention_mask.float().unsqueeze(-1)
    n_real = mask_float.sum().clamp(min=1.0)

    # 1. Per-layer L2 norm of mean-pooled vector
    norms = []
    mean_vecs = []
    for layer_idx in _SELECTED_LAYERS:
        layer = hidden_states[layer_idx]
        mean_vec = (layer * mask_float).sum(dim=0) / n_real
        mean_vecs.append(mean_vec)
        norms.append(mean_vec.norm().unsqueeze(0))

    norms_tensor = torch.cat(norms)  
    # 2. Inter-layer cosine similarities between consecutive selected layers
    cos_sims = []
    for i in range(len(mean_vecs) - 1):
        sim = F.cosine_similarity(
            mean_vecs[i].unsqueeze(0),
            mean_vecs[i + 1].unsqueeze(0),
        )
        cos_sims.append(sim)
    cos_sims_tensor = torch.stack(cos_sims)  # (n_selected - 1,)

    # 3. Log sequence length
    seq_len = torch.tensor([torch.log(n_real + 1.0)])

    # 4. Std of norms across selected layers
    std_norms = torch.tensor([norms_tensor.std()])

    return torch.cat([norms_tensor, cos_sims_tensor, seq_len, std_norms], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """
    Aggregate hidden states and optionally extract geometric features.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
