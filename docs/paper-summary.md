# Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics (EMNLP 2020) - Summary

## Overall Document Summary
This paper introduces **Data Maps**, a model-based visualization and diagnostic tool for large NLP datasets. By leveraging *training dynamics* (how a model's predictions on each training example evolve across epochs), the authors create 2D maps where each example is plotted by:
- **Confidence** (mean probability assigned to the gold label across epochs)
- **Variability** (standard deviation of that probability)

These maps reveal three consistent regions across four datasets (SNLI, MultiNLI, WinoGrande, QNLI):
- **Easy-to-learn** (high confidence, low variability) — the vast majority of examples.
- **Hard-to-learn** (low confidence, low variability).
- **Ambiguous** (high variability).

The paper demonstrates that these regions have distinct roles in model learning, generalization, and data quality.

## Research Purpose
Large datasets have become standard in NLP, but scale makes it hard to assess *quality* rather than just *quantity*. The goal is to automatically characterize the contribution of individual examples to in-distribution (ID) performance, out-of-distribution (OOD) generalization, and overall dataset health — moving from "more data = better" to a nuanced understanding of data utility.

## Research Goal
Construct intuitive, model-dependent coordinates for every training example using only a single training run. Use these coordinates to:
1. Visualize datasets as "maps".
2. Diagnose different regions (easy/hard/ambiguous).
3. Guide better data selection, cleaning, and dataset construction.

## Research Expectations / Hypotheses
- Ambiguous examples (high variability) would be most informative for OOD generalization.
- Easy-to-learn examples dominate datasets and are crucial for optimization but less critical for generalization.
- Hard-to-learn examples often correspond to labeling errors (noise).
- Data maps would be stable across random seeds and reveal similar structures across datasets/tasks/models.

## Quick Summary: How They Achieved It
1. Train a strong model (RoBERTa-large) on the full training set for several epochs, logging the model's predicted probability for the *gold label* of every example at the end of each epoch.
2. For each example compute:
   $$
   \hat{\mu}_i = \frac{1}{E}\sum_{e=1}^{E} p_{\theta^{(e)}}(y_i^* | x_i) \quad \text{(confidence)}
   $$
   $$
   \hat{\sigma}_i = \sqrt{\frac{1}{E}\sum_{e=1}^{E} (p_{\theta^{(e)}}(y_i^* | x_i) - \hat{\mu}_i)^2} \quad \text{(variability)}
   $$
3. Plot every training example in 2D (variability vs. confidence) → the data map.
4. Diagnose regions by:
   - Training models *only* on subsets from each region (33% of data).
   - Injecting artificial noise and re-computing maps.
   - Human evaluation of hard-to-learn examples.
5. Compare against random selection, high-confidence/low-variability subsets, and prior data-selection baselines (forgetting, AFLite, active learning, etc.).

## Key Insights
- **Ambiguous examples** (high variability) give the *best OOD generalization* — often beating the full 100% training set while using only 1/3 of the data.
- **Easy-to-learn examples** form the modal group and are essential for optimization (models fail to converge on tiny ambiguous-only subsets).
- **Hard-to-learn examples** frequently contain labeling errors; a simple linear classifier on confidence scores can surface them with high precision.
- Data maps are surprisingly stable across random seeds and generalize across model architectures (though region membership can shift).
- Shift from quantity to *quality* (via data maps) enables smaller, cleaner, more robust datasets and better OOD performance.
- The method is model-agnostic and cheap (just one training run + simple post-processing).

**Paper takeaway**: Training dynamics are a powerful, under-used signal for understanding what our data actually teaches models.