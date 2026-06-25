# Metaphorical Usage as Contextual Embedding Shift

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12dWOUjs7eKUc2bp9H7w3IB-RIOsa4P0c?usp=sharing)

A small proof-of-concept NLP experiment testing whether metaphorical verb usages show a measurable shift in contextual embedding space.

This project was developed for the course **Figurative Language and NLP** at the University of Tübingen. It is intended as a transparent research MVP rather than a production-ready metaphor detection system.

## Project Overview

The project asks a focused question:

> Do metaphorical uses of a verb occur farther away from the verb's literal-use centroid in contextual embedding space?

The broader theoretical motivation is that figurative language may be better understood as a continuous semantic-pragmatic space rather than as a set of strictly discrete rhetorical categories. For this MVP, the scope is deliberately reduced to **English verb metaphors** and to one operational proxy: the distance between a contextual target-token embedding and a lemma-specific literal centroid.

The main experiment uses **RoBERTa-base** target-token embeddings on a sampled subset of **VUA20 verb instances**. For each target verb, the notebook extracts its contextual embedding, computes lemma-specific literal centroids, and compares literal and metaphorical usages using cosine distance.

## Repository Status

This repository currently represents an MVP / course-project prototype.

- The main notebook runs in **Google Colab**.
- The MVP uses a sampled VUA20 verb subset.
- The current results provide weak-to-moderate support for the contextual embedding shift hypothesis.
- The project should not be interpreted as a complete theory of metaphor or a state-of-the-art metaphor detector.

## Research Question and Hypothesis

### Main research question

Do metaphorical uses of a word occur farther away from its literal-use centroid in contextual embedding space?

### Main hypothesis

For a lemma `w`, compute the literal centroid:

```text
c_lit(w) = mean Enc(x_i, w)
```

where the mean is taken over literal usages of `w`.

For each usage, compute its cosine distance to the literal centroid:

```text
D_lit(x, t) = dist(Enc(x, t), c_lit(t))
```

The prediction is:

```text
D_lit(x_met, t) > D_lit(x_lit, t)
```

This is a weak geometric hypothesis. A positive result suggests that metaphorical usage is associated with contextual shift, not that metaphor can be reduced to cosine distance.

## Method

### Dataset

The MVP uses the Hugging Face dataset:

- `CreativeLang/vua20_metaphor`

The experiment focuses on **verb targets** and keeps only lemmas that have both literal and metaphorical instances.

Current MVP sample:

| Item | Count |
|---|---:|
| Total VUA20 verb instances | 24,930 |
| Literal verb instances | 18,703 |
| Metaphorical verb instances | 6,227 |
| Eligible lemmas with both labels | 784 |
| Selected lemmas | 200 |
| Sampled rows | 1,605 |
| Literal sampled rows | 898 |
| Metaphorical sampled rows | 707 |

### Encoder

The MVP uses:

- Model: `roberta-base`
- Representation: contextual embedding of the annotated target token
- Subword handling: average subword vectors if the target word is split
- Distance metric: cosine distance
- Vector settings:
  - L2 normalization
  - Mean-centering + L2 normalization
  - Remove PC1 + L2 normalization

### Experiments

| Experiment | Purpose | MVP status |
|---|---|---|
| Literal centroid distance | Test whether metaphorical usages are farther from lemma-specific literal centroids | Implemented |
| Lemma-level centroid separation | Explore which lemmas show clearer literal/metaphorical separation | Implemented, exploratory |
| Classification sanity check | Test whether embeddings contain metaphor-relevant signal | Implemented |
| Anisotropy diagnostics | Test whether cosine geometry is affected by common directions | Implemented |
| Polysemy control | Distinguish metaphor-specific shift from ordinary sense variation | Planned |
| MIP-style contextual-isolated gap | Compare contextual target embedding with isolated target-word embedding | Planned |
| SPV-style target-context mismatch | Compare target representation with surrounding context | Planned |

## Current MVP Results

### 1. Literal centroid distance

Metaphorical verb usages are farther from lemma-specific literal centroids across all vector-processing settings.

| Setting | Literal mean | Metaphorical mean | Gap | p-value | Cohen's d |
|---|---:|---:|---:|---:|---:|
| L2 | 0.0549 | 0.0621 | 0.0072 | 2.45e-07 | 0.258 |
| Mean-centered + L2 | 0.4508 | 0.5260 | 0.0751 | 8.77e-15 | 0.387 |
| Remove PC1 + L2 | 0.4600 | 0.5331 | 0.0731 | 1.28e-13 | 0.369 |

**Interpretation:** the result supports a weak version of the contextual embedding shift hypothesis. However, the effect size is small to moderate, so the result should be framed as a geometric tendency rather than proof of a complete metaphor theory.

### 2. Classification sanity check

Simple classifiers perform only slightly above the majority baseline.

| Setting | Classifier | Accuracy | Macro-F1 | Metaphor F1 |
|---|---|---:|---:|---:|
| L2 | Logistic Regression | 0.603 | 0.597 | 0.558 |
| L2 | Linear SVM | 0.595 | 0.591 | 0.555 |
| Mean-centered + L2 | Logistic Regression | 0.600 | 0.595 | 0.556 |
| Remove PC1 + L2 | Logistic Regression | 0.597 | 0.592 | 0.551 |

Majority baseline accuracy: approximately `0.556`.

**Interpretation:** RoBERTa target-token embeddings contain metaphor-relevant signal, but the signal is weak under grouped-by-lemma generalization.

### 3. Anisotropy diagnostics

Raw RoBERTa embeddings are strongly anisotropic. Mean-centering and PC1 removal reduce common-direction effects and make the geometry more interpretable, but they do not clearly improve classification performance.

This means anisotropy correction should be treated as a robustness and interpretability step rather than as a performance-improvement method.

## How to Run

### Option 1: Open in Google Colab

Use the badge at the top of this README, or open the notebook directly in Colab:

```text
https://colab.research.google.com/drive/12dWOUjs7eKUc2bp9H7w3IB-RIOsa4P0c?usp=sharing
```

In Colab, select a GPU runtime:

```text
Runtime -> Change runtime type -> GPU
```

Then run all cells.

### Option 2: Run the repository notebook

Open the notebook from this repository:

```text
notebooks/metaphor_embedding_shift_mvp_vua20_roberta.ipynb
```

The notebook will:

1. load VUA20 from Hugging Face;
2. normalize labels and infer relevant columns;
3. filter to verb targets;
4. sample lemmas with both literal and metaphorical usages;
5. extract target-token embeddings from RoBERTa-base;
6. run distance analysis, classification, and anisotropy diagnostics;
7. save output tables and plots.

Default output directory in Colab:

```text
/content/metaphor_mvp_outputs
```

### Optional local setup

The notebook is primarily designed for Colab. For local reproduction, install the dependencies with:

```bash
pip install -r requirements.txt
```

## Files in This Repository

| File | Purpose |
|---|---|
| `README.md` | Public project introduction and reproduction guide |
| `notebooks/metaphor_embedding_shift_mvp_vua20_roberta.ipynb` | Main Colab MVP notebook |
| `docs/metaphorical_usage_embedding_shift_proposal.pdf` | Short proposal / project write-up |
| `docs/Project_Context_FNLP.md` | Extended project context and iteration plan |
| `requirements.txt` | Optional dependency list for local reproduction |
| `results/` | Aggregate output tables and plots, if included |

## Limitations

- The MVP uses only English verb metaphors.
- It uses only one encoder: RoBERTa-base.
- It uses a sampled subset of VUA20 rather than the full dataset.
- It does not model intended meaning, discourse context, audience interpretation, simile, irony, idiom, or figurative language as a whole.
- The observed embedding shift may reflect ordinary contextual sense variation rather than metaphor-specific geometry.
- Classification performance is only slightly above baseline.
- Lemma-level separation is exploratory because some lemmas have very small sample sizes.

## Next Steps

Planned improvements:

1. Add a polysemy control using literal-literal dispersion.
2. Add a MIP-style contextual-isolated gap measure.
3. Add a SPV-style target-context mismatch measure.
4. Compare raw embeddings, distance features, and combined features in the classifier.
5. Add shuffled-label controls within lemma where possible.
6. Add regression controls for sentence length, subword count, and lemma frequency.
7. Use frame-inspired qualitative analysis to separate metaphorical shift from ordinary polysemy.

## Technical Stack

- Python
- Google Colab
- Hugging Face `datasets`
- Hugging Face `transformers`
- RoBERTa-base
- PyTorch
- scikit-learn
- pandas
- NumPy
- SciPy
- matplotlib

## Data and License Notes

This repository does not redistribute the raw VUA20 dataset. The notebook loads the dataset through Hugging Face during execution.

The code and documentation in this repository are released under the MIT License. Dataset licensing and redistribution terms belong to the original dataset providers.

Generated model caches, checkpoints, and large local Colab outputs should not be committed to version control unless they are small aggregate result files.

## Course Context

This project was developed as a course-project MVP for **Figurative Language and NLP**. It is also part of a broader technical portfolio showing research-oriented NLP prototyping, embedding analysis, experimental design, and error-aware interpretation.

## References

- Leong, C. W., et al. (2020). *A Report on the 2020 VUA and TOEFL Metaphor Detection Shared Task*. Proceedings of the Second Workshop on Figurative Language Processing.
- Steen, G., Dorst, A., Herrmann, J. B., Kaal, A., Krennmayr, T., & Pasma, T. (2010). *A Method for Linguistic Metaphor Identification*. John Benjamins.
