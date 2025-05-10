---
noteId: "6d5efcc02d8411f08e21ed8c3d313aa3"
tags: []

---

# Self-Attention Based Automatic Modulation Classification

This repository contains a reimplementation and extension of the work presented in the paper:  
**"[Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification](https://ieeexplore.ieee.org/document/10042021)"**.

## Overview

The original paper proposes a dual-stream architecture combining both data-driven and expert-driven approaches:

- A **Convolutional Neural Network (CNN)** processes raw IQ samples.
- A **Multilayer Perceptron (MLP)** processes a set of handcrafted features, including:
  - **Instantaneous features**: amplitude, phase, frequency
  - **Statistical features**: high-order moments and cumulants
  - **Spectral features**: spectral power density characteristics, spectrum symmetry, etc.

These two streams are fused via a dedicated **fusion module**, which:
- Projects the concatenated features into a lower-dimensional space using a fully connected layer
- Projects them back to the original space before performing classification

## Contribution

This repository replicates the original architecture and introduces a modification:  
the standard fusion mechanism is replaced with a **self-attention-based module** to improve feature integration and potentially enhance classification performance.

## Reference

If you use this code or idea in your work, please cite the original paper:  
> *Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification*, IEEE, 2023.  
> [Link to paper](https://ieeexplore.ieee.org/document/10042021)
