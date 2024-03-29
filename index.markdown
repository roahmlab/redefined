---
# Front matter. This is where you specify a lot of page variables.
layout: default
title:  "REDEFINED"
date:   2024-03-04 10:00:00 -0500
description: >- # Supports markdown
  Reachability-based Trajectory Design via Exact Formulation of Implicit Neural Signed Distance Functions
show-description: true

# Add page-specifi mathjax functionality. Manage global setting in _config.yml
mathjax: false
# Automatically add permalinks to all headings
# https://github.com/allejo/jekyll-anchor-headings
autoanchor: false

# Preview image for social media cards
# image:
#   path: https://cdn.pixabay.com/photo/2019/09/05/01/11/mountainous-landscape-4452844_1280.jpg
#   height: 100
#   width: 256
#   alt: Random Landscape

# Only the first author is supported by twitter metadata
authors:
  - name: Jonathan Michaux*
    email: jmichaux@umich.edu
  - name: Qingyi Chen*
    email: chenqy@umich.edu
  - name: Challen Enninful Adu
    email: enninful@umich.edu
  - name: Jinsun Liu
    email: jinsunl@umich.edu
  - name: Ram Vasudevan
    email: ramv@umich.edu

# If you just want a general footnote, you can do that too.
# See the sel_map and armour-dev examples.
author-footnotes:
  All authors affiliated with the Department of Robotics of the University of Michigan, Ann Arbor.
  # 1: >- # Supports markdown
  #   You can add random extra footnotes
  # 2: And include websites or emails which are detached from their mailto
  # 3: You can also just specify the email and not have a mailto, or if there's a mailto you want to use, you can specify only that

links:
  - icon: arxiv
    icon-library: simpleicons
    text: ArXiv
    url: https://arxiv.org/abs/2403.12280
  - icon: github
    icon-library: simpleicons
    text: Code
    url: https://github.com/roahmlab/redefined

# End Front Matter
---

---

<!-- BEGIN DOCUMENT HERE -->

{% include sections/authors %}
{% include sections/links %}

---

<!-- # Overview
<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED_overview.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div> -->


<!-- BEGIN ABSTRACT -->
<div markdown="1" class="content-block justify grey">

# Abstract
Generating receding-horizon motion trajectories for autonomous vehicles in real-time while also providing safety
guarantees is challenging. This is because a future trajectory needs to be planned before the previously computed
trajectory is completely executed. This becomes even more difficult if the trajectory is required to satisfy continuous-
time collision-avoidance constraints while accounting for a large number of obstacles. To address these challenges, this
paper proposes a novel real-time, receding-horizon motion planning algorithm named Reachability-based trajectory
Design via Exact Formulation of Implicit NEural signed Distance functions (REDEFINED). REDEFINED first applies
offline reachability analysis to compute zonotope-based reachable sets that overapproximate the motion of the ego
vehicle. During online planning, REDEFINED leverages zonotope arithmetic to construct a neural implicit representation
that computes the exact signed distance between a parameterized swept volume of the ego vehicle and obstacle
vehicles. REDEFINED then implements a novel, real-time optimization framework that utilizes the neural network to
construct a collision avoidance constraint. REDEFINED is compared to a variety of state-of-the-art techniques and is
demonstrated to successfully enable the vehicle to safely navigate through complex environment. Code will be
released upon acceptance of this manuscript.

</div> <!-- END ABSTRACT -->

<!-- BEGIN METHOD -->
<div markdown="1" class="justify">

# Method

![method_figure](assets/redefine_method_final-compressed.png)
{: class="fullwidth"}

<!-- # Contributions -->

</div><!-- END METHOD -->

<!-- START RESULTS -->
<div markdown="1" class="content-block grey justify">

# Simulation Results
The following videos demonstrate the performance of REDEFINED compared to [REFINE](https://roahmlab.github.io/REFINE_website/) under a time limit of 0.35s, 0.30s and 0.25s for each planning iteration. In all the shown cases REFINE is unable to find a feasible solution within the specified time limit so it stops, while REDEFINED is able to constantly find feasible solutions and finally reach the goals.

## 0.35s Time Limit

<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED_0.35s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REDEFINED operating with 0.35s time limit</p>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REFINE_0.35s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REFINE operating with 0.35s time limit</p>
  </div>
</div> 

## 0.30s Time Limit
<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED_0.30s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REDEFINED operating with 0.30s time limit</p>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REFINE_0.30s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REFINE operating with 0.30s time limit</p>
  </div>
</div> 

## 0.25s Time Limit
<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED_0.25s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REDEFINED operating with 0.25s time limit</p>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REFINE_0.25s_scaled.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
    <p align="center">REFINE operating with 0.25s time limit</p>
  </div>
</div> 

</div>  <!-- END RESULTS -->

<!-- START MORE RESULTS -->
<!-- <div markdown="1" class="justify"> 

# More Simulation Results
The following videos demonstrate more results of REDEFINED operating in our MATLAB simulator.

<div class="fullwidth video-container" style="flex-wrap:nowrap; padding: 0 0.2em">
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED-3.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED-7.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
  <div class="video-item" style="min-width:0;">
    <video
      preload="auto"
      autoplay
      controls
      playsinline
      muted
      loop
      style="display:block; width:100%; height:auto; margin-left:auto; margin-right:auto;">
      <source src="assets/REDEFINED-35100.mp4" type="video/mp4">
      Your browser does not support this video.
    </video>
  </div>
</div> 

</div> --> 
<!-- END MORE RESULTS --> 

<div markdown="1" class="justify">
  
# [Related Projects](#related-projects)
  
* [REFINE: Reachability-based Trajectory Design using Robust Feedback Linearization and Zonotopes](https://roahmlab.github.io/REFINE_website/)

</div>

<div markdown="1" class="content-block grey justify">

  # [Citation](#citation)

This project was developed in [Robotics and Optimization for Analysis of Human Motion (ROAHM) Lab](http://www.roahmlab.com/) at University of Michigan - Ann Arbor.

```bibtex
@misc{michaux2024redefined,
      title={Reachability-based Trajectory Design via Exact Formulation of Implicit Neural Signed Distance Functions}, 
      author={Jonathan Michaux and Qingyi Chen and Challen Enninful Adu and Jinsun Liu and Ram Vasudevan},
      year={2024},
      eprint={2403.12280},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
</div>

