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

## Introduction

Reinforcement learning agents are famously easy to reward and famously hard to reward *well*. Give an agent a single scalar signal and, given enough interaction, it will find a way to maximize it <d-cite key="sutton2018reinforcement"></d-cite> — the trouble starts when the thing you actually care about doesn't collapse into one number. Robotic manipulation <d-cite key="intelligence2025pi"></d-cite><d-cite key="todorov2012mujoco"></d-cite><d-cite key="rsoccer"></d-cite>, competitive game-playing <d-cite key="silver2016mastering"></d-cite><d-cite key="silver2018general"></d-cite><d-cite key="berner2019dota"></d-cite><d-cite key="vinyals2019grandmaster"></d-cite>, and aligning language models with human feedback <d-cite key="kaufmann2023survey"></d-cite> all end up juggling several objectives at once, and the standard move is to fold them into one reward with a set of hand-picked weights: $r = \sum_i \lambda_i r_i$.

Those weights are not cosmetic. They define an implicit curriculum <d-cite key="curriculum"></d-cite> — which sub-goal the agent is effectively being pushed toward at any given moment — and getting the curriculum wrong quietly wrecks training. Tilt the weights too far toward whichever component is easiest to farm and the agent will happily overfit to it: a soccer-playing agent rewarded generously for holding onto the ball can become excellent at holding onto the ball and never discover the sparser, harder objective of actually scoring. The usual fix is to sit down and hand-tune $\lambda$ — a slow, brittle, per-environment ritual that has to be redone from scratch the moment the reward structure changes even slightly.

People have tried to get out of this by other doors. Reward shaping <d-cite key="ng1999policy"></d-cite> adds auxiliary feedback to make learning easier, but the priorities it bakes in are static — they don't know or care how far along the agent already is. Curriculum learning <d-cite key="bengio2009curriculum"></d-cite> tackles the staging problem more directly, and automated variants <d-cite key="portelas2020automatic"></d-cite><d-cite key="li2023understanding"></d-cite><d-cite key="graves2017automated"></d-cite> — including self-paced learning, where the model's own performance decides what to emphasize next <d-cite key="kumar2010self"></d-cite> — go a long way toward removing the hand-crafting. But almost all of that machinery is built to sequence *tasks* or *environments* <d-cite key="xiao2025collaborative"></d-cite><d-cite key="lv2026metagrasp"></d-cite><d-cite key="mead2026multi"></d-cite>, not the components sitting inside a single reward. Hand-staged curricula over reward terms specifically remain a bespoke, per-domain exercise <d-cite key="xiao2023flying"></d-cite><d-cite key="lian2026curriculum"></d-cite><d-cite key="efendi2026learning"></d-cite>. Meanwhile, multi-objective RL methods that decompose the reward — Q-decomposition keeps one value function per component and coordinates them through a shared decision rule <d-cite key="russell2003q"></d-cite><d-cite key="vanseijen2017hybrid"></d-cite><d-cite key="fatemi2022orchestrated"></d-cite>, and methods like Envelope Q-learning or GPI-based prioritization <d-cite key="yang-envelope"></d-cite><d-cite key="alegre2023sample"></d-cite> push the idea further — but they're usually aimed at recovering a whole Pareto front of trade-off policies, not at handing back one well-shaped policy for a single deployment.

That's the gap DyLam sits in: nobody was treating the *weights themselves* as the curriculum. The idea is to keep the reward decomposed — one component, one signal — and let a competence-style measure decide, online, how much attention each component deserves right now, borrowing the "responsibility signal" intuition from Multiple Model-Based RL <d-cite key="doya2002multiple"></d-cite>. Concretely, DyLam tracks each component's recent return against a rough estimate of where that component saturates, and reweights toward whichever components are still underperforming while quietly turning down the ones the agent has already mastered. No fixed schedule, no per-environment retuning — just two numbers per component, a floor and a ceiling on its return, and the weights find their own way over the course of training. An interactive walkthrough of this update over one slice of real training data follows next, before the formal statement of the mechanism.

<!-- BEGIN #43: lambda-simplex scrubber Embed (results). Provisional
     placement -- issue #39's assembly ticket moves this into a full
     "curriculum results" subsection once that prose exists. Please keep
     this block intact when adding the Mechanism section elsewhere in this
     file, to minimize merge conflicts. -->

## Results

DyLam decomposes a multi-objective reward into per-Component critics and adapts a weight $\lambda(t)$ over training to trade them off. Scrub the slider below to watch $\lambda$ move through training on ChickenBanana and see the greedy policy it induces at each moment. Click anywhere in the weight simplex to try a different, fixed weighting against the *same* learned Component Q-tables at that same moment -- the counterfactual this Embed exists to make checkable.

<div class="l-page">
  <iframe src="{{ 'assets/html/submission/lambda_simplex_scrubber.html' | relative_url }}" frameborder="0" scrolling="no" height="720" width="100%" style="border: none;"></iframe>
</div>

Left: the ChickenBanana grid, with the currently-displayed policy's greedy action drawn in each reachable cell (before either pickup) and its rollout path overlaid statically -- the path is not animated, so two weightings can be compared at a glance by switching between them. Middle: the three-Component weight simplex, coloured by the behaviour class each weighting's policy achieves at the currently selected episode, with DyLam's own $\lambda$ trajectory drawn as a trail up to that episode. Clicking elsewhere fixes that weighting (free mode, amber) so the time slider keeps scrubbing the counterfactual; the button above returns to DyLam's own trajectory (follow mode, green). Right: the rollout's per-Component returns (solid bars) against the method's own r_max ceilings (dashed), with DyLam's actual training-time return at that episode ghosted behind for reference.

<!-- END #43 -->
