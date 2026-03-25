# Quantifying Information Gain in SIREN Prepatterning

## Why this experiment
We train a modulated SIREN and an NCA end-to-end:
- SIREN maps coordinates (plus latent) to an initialization field.
- NCA refines that field through local updates (about 10-15 steps).

So the core question is:

How much useful information is already present in the SIREN prepattern, before NCA refinement?

![Siren prepattern](/assets/siren/siren-examples_best.png)
![Final patterns](/assets/siren/examples_best.png)

Prepatterns contain coarse global structure and symmetry breaks. Final outputs add specificity, local detail and cleanup.

## Why cohort/relative metrics
The key design choice here is to avoid absolute, single-image scores as the main story.

Instead, we focus on **cohort-relative metrics**:
- Compare images to other images in the same stage (`pre` cohort, `target` cohort).
- Track how those internal relationships change from `pre` to `target`.

Why this matters:
- The model is trained on a set of patterns, not one image in isolation.
- Many biologically inspired effects are relational: symmetry classes, motif reuse, and controlled diversification.
- We would not know what a "good" MSE between a prepattern and a target would be. But the relative change between prepattern and post-NCA is well defined.

In practice, we summarize pairwise relations (upper triangle of pairwise matrices) and compare stage-level distributions.

## Metrics we implemented

### 1. Internal pairwise SSIM (within cohort)
Definition:
- Compute SSIM for all image pairs within a stage (`pre` or `target`), then summarize.

What it captures:
- Perceptual/structural similarity across samples. As the images  diversify post-NCA, internal SSIM will hopefully decrease.

Strengths:
- Easy to interpret.
- Sensitive to spatial structure (edges/shapes).
- Directly cohort-relative: asks how the population geometry changes after NCA.

Weaknesses:
- A relative drop says nothing about how good the. Random noise would give a big drop.
- Not an information-theoretic quantity.

### 2. Internal pairwise NMI in pixel space
Definition:
- Histogram-based normalized MI on pixel intensities, pairwise within stage.

What it captures:
- Shared intensity statistics.

Strengths:
- Classical information metric.
- Robust to monotonic intensity transforms.
- Works naturally as a relative cohort statistic.

Weaknesses:
- Weak spatial sensitivity; histogram overlap can stay high even when geometry differs.
- Can be dominated by large uniform backgrounds.

## Fourier-space extension
To make MI more spatially meaningful, we compute the same cohort MI idea in Fourier magnitude space.

![image from assets](/assets/siren/fourier_metrics1.png)

### 3. Internal pairwise Fourier NMI
Definition:
- NMI between log-magnitude spectral distributions (pairwise, within stage).

What it captures:
- Similarity of spatial frequency structure rather than raw pixel histograms.

Strengths:
- More sensitive to edges, texture scale, and shape frequency content.
- Less fooled by flat backgrounds than pixel-histogram NMI.
- Retains the same cohort-relative comparison framework.

Weaknesses:
- Phase-insensitive; two images with similar spectra but different arrangement can look very different.
- Still a global summary, not local correspondence.

## Gregor-style positional identifiability
Inspired by Thomas Gregor-style decoding logic:
- Ask whether local signal uniquely identifies position.
- Operationalized as MI between discretized RGB code and pixel position, plus a pixel uniqueness map.

![gregor-style](/assets/siren/gregor-style.png)

### 4. Positional MI and uniqueness
Definition:
- $I(\text{RGB}; \text{position})$ per image.
- Uniqueness score per pixel: inverse frequency of its RGB code.

What it captures:
- How strongly color/state encodes where you are in the canvas.

Strengths:
- Directly tied to your biological/developmental intuition.
- Gives both scalar summary and interpretable maps.
- Still compatible with cohort-relative analysis when summarized across the set.

Weaknesses:
- Depends on color binning choice.
- Can overestimate identifiability if tiny color noise creates artificial uniqueness.

## Current takeaways
- **Main point:** the best signal came from **relative cohort structure**, not absolute scores.
- **SSIM:** useful for tracking structural cohesion/diversification across the cohort.
- **Pixel NMI:** still informative, but often too coarse in image space.
- **Fourier NMI:** better relational structural signal than pixel NMI here.
- **Gregor positional MI:** closest to the mechanistic question "does prepattern encode position?"

In short: no single metric is enough. The most honest read is a **multi-metric, cohort-relative** story.

## Role of noise baseline (briefly)
Structured noise is used as a sanity-check floor, not the main axis of interpretation.

![structured noise](/assets/siren/noise1.png)

It helps detect degenerate metrics, but conclusions are driven by `pre` vs `target` cohort geometry.

## Next ideas
- Foreground-masked versions of all metrics (remove white background bias).
- Local/windowed positional MI to separate coarse global layout from fine detail.
- Decoder-based probe: train a small classifier/regressor to predict position from local state.
