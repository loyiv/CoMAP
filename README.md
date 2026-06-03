# CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents

<p align="center">
  <a href="https://arxiv.org/abs/2606.02372">
    <img src="https://img.shields.io/badge/arXiv-2606.02372-b31b1b.svg" alt="arXiv">
  </a>
  <a href="comap/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="License">
  </a>
</p>

<p align="center">
  <a href="https://loyiv.github.io/">Youwei Liu</a>,
  <a href="https://iwangjian.github.io/">Jian Wang†</a>,
  <a href="https://wanghanlinhenry.github.io/">Hanlin Wang</a>,
  <a href="https://www4.comp.polyu.edu.hk/~cswjli/">Wenjie Li</a>
</p>

## Overview

<p align="center">
  <img src="assets/figure1.png" width="85%">
</p>

**Figure 1. Conceptual illustration of CoMAP.**  
CoMAP co-evolves the world model and the agent policy in a closed loop. The world model provides lookahead states for policy improvement, while the agent policy generates on-policy interactions for world-model updating.

<p align="center">
  <img src="assets/fiugre2.png" width="95%">
</p>

**Figure 2. Framework overview.**  
At each step, the agent first drafts an action, the world model predicts its future state, and the policy performs future-aware reflection to refine the action. The resulting trajectories are used for on-policy self-distillation of the world model and policy-side evolution.

---


## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{liu2026comap,
  title   = {Co-Evolving World Models and Agent Policies for LLM Agents},
  author  = {Liu, Youwei and Wang, Jian and Wang, Hanlin and Li, Wenjie},
  journal = {arXiv preprint arXiv:2606.02372},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.02372}
}
```

---

## Contact

For questions, please contact:

```text
loyiv5477@gmail.com
```
