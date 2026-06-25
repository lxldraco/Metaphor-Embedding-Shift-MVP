# Project Context: Metaphorical Usage as Contextual Embedding Shift

## 1. Project Identity

**Course:** Figurative Language and NLP  
**Credits:** 6 ECTS  
**Current title:** *Metaphorical Usage as Contextual Embedding Shift: A Small Proof of Concept*  
**Current scope:** Scope 1 — metaphor only  
**Main research question:** Do metaphorical uses of a word occur farther away from its literal-use centroid in contextual embedding space?

This project is a focused computational linguistics term project. It began from a broader theoretical proposal, *Figurative Language as a Continuous Semantic Space*, but has now been deliberately reduced to a smaller and more testable problem: whether contextual embeddings can capture a systematic representational shift between literal and metaphorical usages of the same lexical item. The project should therefore be treated as a theory-driven proof of concept, not as a full metaphor detection system and not as a complete proof of the figurative-language continuum hypothesis.

The central operational idea is simple: if metaphor involves a contextual semantic shift, then a target word used metaphorically may be represented farther away from the center of its literal usages than another literal usage of the same word. This turns the earlier abstract formulation of semantic-pragmatic distance into an executable embedding-space experiment.

## 2. Background and Motivation

The original theoretical motivation was that figurative language may be better understood as a continuous semantic-pragmatic space than as a set of strictly discrete categories. Literal language, conventionalized metaphor, novel metaphor, idiom, simile, irony, sarcasm, and humor may overlap, and many expressions do not fit cleanly into a single rhetorical category. However, this full framework is too broad for a controlled course project.

The current project keeps only the part that can be operationalized with existing human-annotated metaphor data: metaphorical usage as contextual embedding shift. In this reduced form, the project asks whether the contextual representation of a target token changes in a measurable way when the token is used metaphorically rather than literally.

This scope directly responds to earlier feedback: the original proposal was too broad; the encoding function `Enc()` was not sufficiently explicit; transformer embeddings may suffer from anisotropy; and the dataset should be a real human-annotated corpus rather than an LLM-generated dataset. The current design addresses these points by using a clearly defined target-token embedding, a within-lemma comparison design, and VUA20 verb instances.

The newest iteration is also inspired by *FrameBERT: Conceptual Metaphor Detection with Frame Embedding Learning*. FrameBERT uses RoBERTa to obtain three relevant representations: sentence representation, contextual target-word representation, and isolated target-word representation. It then uses MIP to model the gap between contextual and isolated target meaning, SPV to model the contrast between target word and sentential context, and FrameNet embeddings to add explicit conceptual-frame information. This project does not attempt to reproduce full FrameBERT. Instead, it adopts a minimal FrameBERT-inspired extension: add MIP-style and SPV-style distance diagnostics and use frame-based qualitative analysis as a control for metaphor vs. ordinary polysemy.

## 3. Scope Limits

The project focuses only on metaphorical usages without expanding to other figurative devices. Simile, irony, sarcasm, idiom, proverb, humor, hyperbole, metonymy, Chinese chengyu, and broader discourse-level figurative interpretation are excluded from the main experiment. Simile is especially excluded because explicit comparison markers such as *like* and *as* introduce a different computational problem from metaphor without explicit comparison markers.

The project also does not attempt to compare a literal meaning representation `Enc(S(x))` with an intended meaning representation `Enc(I)`. Most real metaphor datasets do not provide full literal paraphrases and intended meanings. Therefore, the current project does not claim to directly measure semantic-pragmatic distance between surface meaning and intended meaning. Instead, it measures a more modest and data-compatible proxy: the position of a target token in contextual embedding space.

The project does not train a new large model, does not pursue state-of-the-art metaphor detection, does not reproduce full FrameBERT, and does not use LLM-generated data as primary evidence. Classification is included only as a sanity check for whether the embeddings and theory-driven distance features contain metaphor-relevant signal.

## 4. Operational Definition of `Enc()`

In this project,

$$
Enc(x,t)=\text{contextual embedding of target token }t\text{ in sentence }x.
$$

Here, `x` is the sentence containing the target item, and `t` is the target word/token annotated as literal or metaphorical. `Enc(x,t)` is extracted from a pretrained transformer encoder, currently RoBERTa-base. If the target word is split into multiple subword tokens, the subword vectors are averaged. The resulting target-token vector is then L2-normalized before cosine distance is computed.

The main representation level is the target-token contextual embedding. The new iteration additionally extracts two lightweight auxiliary representations: an isolated target-word embedding `Enc(t)` for the MIP-style gap, and a context representation derived from the sentence or from mean pooling over non-target tokens for the SPV-style mismatch. The project compares different usages of the same lemma, not unrelated words across the whole embedding space.

## 5. Core Hypotheses

### H1: Literal centroid distance

For each lemma `w`, compute a literal-use centroid:

$$
c_{\text{lit}}(w)=\frac{1}{N}\sum_{i=1}^{N}Enc(x_i,w),
$$

where the sum ranges over literal usages of lemma `w`. For each usage, compute its cosine distance to the literal centroid:

$$
D_{\text{lit}}(x,t)=dist(Enc(x,t),c_{\text{lit}}(t)).
$$

The main prediction is:

$$
D_{\text{lit}}(x_{\text{met}},t)>D_{\text{lit}}(x_{\text{lit}},t).
$$

This hypothesis should be interpreted weakly: metaphorical usages are expected to show a measurable contextual shift away from literal usages. The project must not claim that metaphor is reducible to cosine distance.

### H2: MIP-style contextual--isolated gap

Inspired by FrameBERT's use of MIP, the next iteration adds a diagnostic that compares the target word in context with the target word in isolation:

$$
D_{\text{MIP}}(x,t)=dist(Enc(x,t),Enc(t)).
$$

The prediction is:

$$
D_{\text{MIP}}(x_{\text{met}},t)>D_{\text{MIP}}(x_{\text{lit}},t).
$$

This tests whether metaphorical usages deviate more strongly from a basic or isolated lexical representation.

### H3: SPV-style target--context mismatch

Inspired by FrameBERT's use of SPV, the next iteration adds a target--context mismatch score. The context representation can be the sentence representation or, preferably, mean pooling over non-target token vectors:

$$
D_{\text{SPV}}(x,t)=dist(v_{target}(x,t),v_{context}(x)).
$$

The prediction is:

$$
D_{\text{SPV}}(x_{\text{met}},t)>D_{\text{SPV}}(x_{\text{lit}},t).
$$

This tests whether metaphorical targets are less semantically cohesive with their surrounding context than literal targets.

### H4: Classification sanity check

If contextual embeddings and theory-driven distances contain metaphor-relevant information, then simple classifiers should perform above a majority baseline. The next iteration should compare three input types: raw target embedding, distance features only, and raw embedding plus distance features. The relevant distance features are literal-centroid distance, MIP-style gap, and SPV-style mismatch. This remains a sanity check, not a state-of-the-art detector.

### H5: Polysemy / frame qualitative control

The observed embedding shift may reflect general sense variation rather than metaphor-specific shift. Therefore, the next iteration should compare metaphorical shift with ordinary literal-literal dispersion and, if possible, literal sense clusters. A possible expected pattern is:

$$
D_{\text{same literal sense}} < D_{\text{different literal senses}} \leq D_{\text{literal-metaphor}}.
$$

Frame-inspired qualitative analysis is the project's main innovation point. The goal is to examine whether high-distance literal cases are ordinary polysemy, while metaphorical cases involve a stronger mismatch between a target word's basic frame and its contextual frame. If metaphorical distances are not clearly larger than ordinary polysemy distances, the result should be framed as evidence for general contextual sense shift rather than metaphor-specific geometry.

### H6: Anisotropy robustness

Transformer embeddings are anisotropic, so raw cosine distances may be affected by common directions and other representation artifacts. The project therefore treats cosine distance as an operational approximation rather than a direct measure of meaning. The main robustness settings are L2 normalization, mean-centering followed by L2 normalization, and optional removal of top principal components. This is now secondary to the MIP/SPV and polysemy/frame analyses.

## 6. Dataset and Current MVP Setup

The current MVP uses VUA20 verb instances only. This is appropriate because VUA20 provides human metaphor annotations, token-level labels, and enough verb data to support within-lemma comparisons. The MVP selected only lemmas that contain both literal and metaphorical examples.

Current MVP statistics:

| Item | Value |
|---|---:|
| Total VUA20 verb subset | 24,930 |
| Literal verb instances | 18,703 |
| Metaphorical verb instances | 6,227 |
| Eligible lemmas with both labels | 784 |
| Selected lemmas | 200 |
| Sampled rows | 1,605 |
| Literal sampled rows | 898 |
| Metaphorical sampled rows | 707 |

The setup is valid for Scope 1 because it focuses on within-lemma literal/metaphorical contrast rather than global metaphor detection across unrelated lexical items. VUA20 is also compatible with the FrameBERT-inspired adjustment because FrameBERT itself evaluates on VUA20, MOH-X, and TroFi, and uses RoBERTa-based representations similar to the current MVP backbone.

## 7. MVP v1.0 Results and Interpretation

The MVP provides weak-to-moderate support for the contextual embedding shift hypothesis. In Experiment 1, metaphorical verb usages were significantly farther from lemma-specific literal centroids than literal usages across all representation settings:

| Setting | Literal mean | Metaphorical mean | Gap | p-value | Cohen's d |
|---|---:|---:|---:|---:|---:|
| L2 | 0.0549 | 0.0621 | 0.0072 | 2.45e-07 | 0.258 |
| Mean-centered + L2 | 0.4508 | 0.5260 | 0.0751 | 8.77e-15 | 0.387 |
| Remove PC1 + L2 | 0.4600 | 0.5331 | 0.0731 | 1.28e-13 | 0.369 |

These results support a geometric tendency: metaphorical usages tend to be farther away from literal centroids. However, the effect size is small to moderate, so the result should not be presented as proof of a complete metaphor theory.

The original lemma-level centroid separation analysis showed that some lemmas have clear separation, but many top-ranked cases had very small sample sizes, sometimes only one literal and one metaphorical instance. This analysis should be retained mainly as a source of qualitative examples and should use a minimum sample threshold where possible.

The original classification sanity check found that simple classifiers performed only slightly above the majority baseline. The best logistic regression setting reached approximately 0.603 accuracy and 0.597 macro-F1, compared with a majority baseline of 0.556. This suggests that RoBERTa target-token embeddings contain metaphor-relevant signal, but the signal is weak under grouped-by-lemma generalization.

The anisotropy check confirmed strong anisotropy in raw RoBERTa embeddings. Mean-centering and PC removal substantially improved geometric interpretability by reducing common-direction effects. However, anisotropy reduction did not significantly improve classification performance. This means anisotropy correction should be framed as a robustness and interpretability tool, not as a performance booster.

## 8. Current Experiment Structure

The next version of the project should use the following structure:

| Experiment | Purpose | Status |
|---|---|---|
| Exp 1: Literal centroid distance | Test the main hypothesis | Keep as main experiment |
| Exp 2: MIP-style contextual--isolated gap | Test basic meaning deviation | Add |
| Exp 3: SPV-style target--context mismatch | Test contextual mismatch | Add |
| Exp 4: Classification sanity check | Compare raw embeddings, distance features, and combined features | Modify lightly |
| Exp 5: Polysemy / frame qualitative control | Distinguish metaphor shift from ordinary sense variation | Add and prioritize |
| Exp 6: Anisotropy robustness | Test whether geometry is stable under normalization | Keep but downgrade |

Additional lightweight controls should include shuffled-label tests and a surface-factor regression:

```text
distance ~ label + sentence_length + subword_count + lemma_frequency
```

These controls help test whether the metaphor label still predicts distance after accounting for simple surface factors.

## 9. Working Interpretation for the Project Report

The safest project-level conclusion is:

> The MVP supports a weak version of the contextual embedding shift hypothesis: metaphorical verb usages are, on average, farther from their lemma-specific literal centroids than literal usages. However, the effect is small to moderate, classification performance remains only slightly above baseline, and the result should be interpreted as evidence for a geometric tendency rather than as proof that metaphor is directly encoded as embedding distance.

The FrameBERT-inspired iteration refines this interpretation. Instead of asking only whether metaphor is farther from a literal centroid, the next version asks whether metaphorical usage simultaneously shows literal-centroid shift, contextual--isolated deviation, and target--context mismatch. The polysemy/frame control then asks whether this pattern is stronger for metaphor than for ordinary sense variation.

## 10. Rejected or Deferred Directions

The following directions are intentionally outside the current project scope:

- full figurative language continuum modeling;
- simile, irony, sarcasm, humor, idiom, proverb, hyperbole, or metonymy;
- discourse-level and audience-level context modeling;
- human participant experiments;
- new metaphor annotation;
- LLM-generated corpus construction;
- full `Enc(S(x))` vs. `Enc(I)` comparison;
- reproducing full FrameBERT or training a FrameNet conceptual encoder;
- training a new metaphor-aware model;
- pursuing state-of-the-art metaphor detection performance.

These topics may appear in the introduction or future work, but they should not be reintroduced into the main experimental design.

## 11. Immediate Next Steps

1. Add the MIP-style contextual--isolated gap.
2. Add the SPV-style target--context mismatch.
3. Modify the classifier to compare raw embeddings, distance features, and combined features.
4. Add the polysemy / frame qualitative control as the main innovation point.
5. Add shuffled-label control within lemma where possible.
6. Keep anisotropy robustness as a secondary check.
7. Add regression control for sentence length, subword count, and lemma frequency.
8. Rewrite the proposal/report around the reduced but FrameBERT-informed Scope 1 design.

## 12. Final Project Framing

The project should be framed as a small, transparent, theory-driven computational experiment. Its contribution is not a new detector, but a controlled test of whether contextual embedding geometry reflects a metaphor-relevant shift within the same lemma. A positive result supports a weak semantic-distance intuition. A mixed or negative result is still valuable because it clarifies the limits of raw contextual embeddings, the importance of anisotropy, and the need to separate metaphor from ordinary polysemy. The FrameBERT-inspired additions make the project stronger without changing its scale: they connect the MVP to MIP, SPV, and conceptual-frame analysis while preserving the minimal proof-of-concept design.
