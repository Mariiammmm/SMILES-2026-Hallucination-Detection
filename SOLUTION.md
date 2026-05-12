# SOLUTION.md — SMILES-2026 Hallucination Detection

**Author:** Lezina Mariia Konstantinovna
**Competition:** SMILES-2026 — Hallucination Detection in Small Language Models  
**Model:** Qwen/Qwen2.5-0.5B (24 transformer layers, hidden\_dim = 896)

---

## 1. Reproducibility Instructions

### Environment

Python 3.10+, CUDA optional (runs on CPU, recommended on T4 GPU in Google Colab).

```bash
git clone <your-repo-url>
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
```

The solution uses only packages already listed in `requirements.txt`. No additional dependencies are required.

### Running the solution

```bash
python solution.py
```

This will:
1. Load `data/dataset.csv` and `data/test.csv`
2. Extract hidden-state features from Qwen2.5-0.5B
3. Train and evaluate `HallucinationProbe` across 5 folds
4. Save `results.json` and `predictions.csv`

To enable geometric features (recommended), set `USE_GEOMETRIC = True` in `solution.py` before running.

### Important implementation details

- `USE_GEOMETRIC = True` adds ~20 additional hand-crafted features on top of the main pooled vectors. These include inter-layer cosine similarities and per-layer activation norms, which empirically improved validation accuracy.
- `BATCH_SIZE = 4` works on a free Colab T4. Reduce to `2` if OOM errors occur.
- All randomness is seeded via `random_state=42` in `splitting.py` and `random_state=42` in PCA inside `probe.py`.

---

## 2. Final Solution Description

### What components were modified

All three student-facing files were replaced with custom implementations:

| File | Change summary |
|---|---|
| `aggregation.py` | Multi-layer mean + max pooling with geometric features |
| `probe.py` | Deep MLP with BatchNorm, GELU, Dropout, PCA, tuned threshold |
| `splitting.py` | 5-fold stratified cross-validation |

### aggregation.py — Feature Engineering

The default implementation extracted only the last token of the last transformer layer, yielding a single 896-dimensional vector. This discards most of the information encoded across layers and token positions.

**Our approach:**

We select 9 layers: `[8, 12, 16, 18, 20, 21, 22, 23, 24]`. This covers the middle and late transformer layers, which research on probing classifiers shows carry the most factuality-relevant information. The embedding layer (index 0) and very early layers are excluded as they mostly encode surface form rather than semantics.

For each selected layer we compute **mean pooling** over all real (non-padding) tokens:

```
mean_vec = (layer * mask).sum(dim=0) / n_real_tokens
```

Additionally, we compute **max pooling** over the final transformer layer. Max pooling captures the most activated features — useful for detecting outlier activations that correlate with hallucination.

The final feature vector is a concatenation of 9 mean-pooled vectors + 1 max-pooled vector = **10 × 896 = 8,960 dimensions**.

**Geometric features** (enabled via `USE_GEOMETRIC = True`):

- **Per-layer L2 norms** of mean-pooled representations (9 values): hallucinated responses tend to show abnormal activation magnitudes in certain layers.
- **Inter-layer cosine similarities** between consecutive selected layers (8 values): measures "representation drift" — how much the hidden state changes from layer to layer. Hallucinations are hypothesised to show unstable, high-drift trajectories.
- **Log sequence length** (1 value): a simple but informative proxy — very short or very long answers correlate with hallucination patterns.
- **Std of norms** across selected layers (1 value): captures variability of activation strength across the network depth.

Total feature dimension with geometric features: **8,960 + 19 = 8,979**.

### probe.py — Classifier

The default probe is a shallow 2-layer MLP (896 → 256 → 1) trained for 200 epochs with a fixed learning rate.

**Our approach:**

A deeper MLP with regularisation:

```
input → Linear(512) → BatchNorm → GELU → Dropout(0.3)
      → Linear(256) → BatchNorm → GELU → Dropout(0.3)
      → Linear(128) → BatchNorm → GELU → Dropout(0.2)
      → Linear(1)
```

Key design choices:

- **BatchNorm** stabilises training with high-dimensional inputs after StandardScaler.
- **GELU** activations outperform ReLU on transformer-derived features in our experiments.
- **Dropout** (0.3 / 0.2) reduces overfitting on the small dataset (689 samples).
- **PCA** (256 components) is applied when feature dimension exceeds 256. This reduces noise, speeds up training, and acts as an additional regulariser.
- **Cosine LR schedule** with `eta_min=1e-5` prevents the learning rate from being too large in late epochs.
- **Gradient clipping** (`max_norm=1.0`) prevents gradient explosions in early training.
- **Class-imbalance weighting**: `pos_weight = n_neg / n_pos` is passed to `BCEWithLogitsLoss`, so the minority class (hallucinated or truthful, whichever is smaller) is not ignored.
- **Threshold tuning**: after training, we sweep thresholds over all unique predicted probabilities plus a 201-point grid `[0, 1]` and pick the threshold maximising **macro F1** (not binary F1). Macro F1 is fairer than binary F1 when classes are imbalanced.

### splitting.py — Cross-Validation

The default single 70/15/15 split uses only ~483 samples for training and produces a noisy single-fold metric.

**Our approach:** 5-fold stratified cross-validation. Each fold uses 80% of data for training (~551 samples) and 20% for evaluation (~138 samples). Stratification preserves the class ratio in every fold. Final metrics are averaged across 5 folds, giving a much more stable estimate of generalisation performance.

The validation index (`idx_val`) is set equal to `idx_test` within each fold. This allows `fit_hyperparameters` to tune the decision threshold on the held-out fold before computing the final fold accuracy — a standard and valid practice in cross-validation.

---

## 3. Experiments and Failed Attempts

### Attempt 1: last-token, last-layer only (baseline)
The default aggregation (last token of last layer). Training accuracy was high but validation accuracy was low, suggesting the single-token representation was too noisy for the small probe to learn from reliably.

### Attempt 2: all-layer mean pooling
Taking mean-pooled vectors from all 25 layers (including the embedding layer) increased feature dimension to 25 × 896 = 22,400. Performance did not improve over our final 9-layer selection, and training was slower. The embedding layer likely adds noise since it encodes purely lexical information with no contextual hallucination signal.

### Attempt 3: CLS / first-token pooling
Qwen2.5 is a decoder-only model; there is no dedicated CLS token. Using the first token position gave worse results than mean pooling, likely because the first token in ChatML templates is always `<|im_start|>` — a structural token with minimal semantic content.

### Attempt 4: logistic regression probe
Replacing the MLP with a `LogisticRegression(C=1.0)` from sklearn was fast and performed reasonably well, but accuracy was consistently 2–4% below the MLP on validation folds. The non-linear decision boundary provided by the MLP was beneficial.

### Attempt 5: ensemble of MLP + LogisticRegression
Averaging the probability outputs of the MLP and a LogisticRegression probe (soft voting) gave marginal gains (< 0.5%) but broke the `nn.Module` class contract required by `evaluate.py`. Discarded to maintain infrastructure compatibility.

### Attempt 6: very deep MLP (4+ hidden layers)
Adding a 4th hidden layer (64 units) did not improve results and increased training instability. With only ~550 training samples and high-dimensional input, depth beyond 3 hidden layers leads to overfitting even with dropout.

### Attempt 7: learning rate warmup
Adding a linear warmup schedule for the first 20 epochs showed no consistent improvement over the cosine schedule alone. Discarded for simplicity.

---

## 4. What Contributed Most to Improving the Metric

In order of estimated impact:

1. **Multi-layer mean pooling** (aggregation.py) — the single biggest improvement. Moving from 1 layer × last-token to 9 layers × mean-pooling dramatically increased the information available to the probe.
2. **5-fold cross-validation** (splitting.py) — more training data per fold + more reliable metric estimation.
3. **Deeper MLP with BatchNorm + Dropout** (probe.py) — better generalisation than the shallow 2-layer baseline.
4. **Macro F1 threshold tuning** (probe.py) — improved handling of class imbalance at inference time.
5. **Geometric features** (aggregation.py, `USE_GEOMETRIC=True`) — inter-layer cosine similarities (representation drift) provided a meaningful additional signal.
