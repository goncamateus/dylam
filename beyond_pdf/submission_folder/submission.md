---
layout: distill
title: DyLam
description: Learning with Decomposed and Dynamically Prioritized Reward Signals
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous
    affiliations:
      name: Anonymous

# Only add author names for camera-ready
# authors:
#   - name: Author Name
#     url: "https://[url_of_author]"
#     affiliations:
#       name: Research Center, University Name

# Must be the same name as your submission. Do not change this name, just use "submission.bib".
bibliography: submission.bib

# Add a table of contents to your submission.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the submission to work correctly.
toc:
  - name: Introduction
  - name: Mechanism
  - name: Results
  - name: Ablation
---

<!-- BEGIN #43: lambda-simplex scrubber Embed (results). Provisional
     placement -- issue #39's assembly ticket moves this into a full
     "curriculum results" subsection once that prose exists. Please keep
     this block intact when adding the Introduction/Mechanism sections
     elsewhere in this file, to minimize merge conflicts. -->

## Results

DyLam decomposes a multi-objective reward into per-Component critics and adapts a weight $\lambda(t)$ over training to trade them off. Scrub the slider below to watch $\lambda$ move through training on ChickenBanana and see the greedy policy it induces at each moment. Click anywhere in the weight simplex to try a different, fixed weighting against the *same* learned Component Q-tables at that same moment -- the counterfactual this Embed exists to make checkable.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/lambda_simplex_scrubber.html' | relative_url }}" frameborder="0" scrolling="no" height="720" width="100%" style="border: none;"></iframe>
</div>

Left: the ChickenBanana grid, with the currently-displayed policy's greedy action drawn in each reachable cell (before either pickup) and its rollout path overlaid statically -- the path is not animated, so two weightings can be compared at a glance by switching between them. Middle: the three-Component weight simplex, coloured by the behaviour class each weighting's policy achieves at the currently selected episode, with DyLam's own $\lambda$ trajectory drawn as a trail up to that episode. Clicking elsewhere fixes that weighting (free mode, amber) so the time slider keeps scrubbing the counterfactual; the button above returns to DyLam's own trajectory (follow mode, green). Right: the rollout's per-Component returns (solid bars) against the method's own r_max ceilings (dashed), with DyLam's actual training-time return at that episode ghosted behind for reference.

<!-- END #43 -->
