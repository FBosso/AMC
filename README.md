# Self-Attention Based Automatic Modulation Classification  
<a href="https://mapflap.github.io"><img align="right" src="https://i.ibb.co/NspjQxV/logo.webp" alt="logo" width="220"></a>

## Overview

*This repository contains a reimplementation and extension of a deep learning framework for automatic modulation classification, inspired by the paper [Toward Next-Generation Signal Intelligence](https://ieeexplore.ieee.org/document/10042021). The original architecture combines a CNN for raw IQ samples with an MLP for handcrafted features, which are fused before classification. This project replicates the original setup and replaces the fusion mechanism with self-attention and cross-attention modules to enhance feature integration.*

## Author  
[Bosso Francesco](https://github.com/FBosso) – (fra.bosso97@gmail.com)

## Problem statement

Automatic modulation classification is a fundamental task in signal intelligence, aiming to identify the modulation scheme of a given radio signal. Traditional methods often rely exclusively on expert-crafted features or purely data-driven approaches, each with its own limitations in terms of flexibility and generalization.

The architecture reproduced in this project combines both perspectives by processing raw IQ samples through a convolutional neural network (CNN) and handcrafted features through a multilayer perceptron (MLP). These two streams are then fused to perform classification. In the original work, fusion was handled by projecting the concatenated features into a latent space using a dense layer and then reconstructing the feature space prior to classification.

This implementation maintains the dual-stream encoder design but replaces the dense fusion layer with two alternative mechanisms: self-attention and cross-attention. In the cross-attention setup, the handcrafted features are used as the `Query` while the raw IQ data act as `Key` and `Value`. These attention-based strategies allow for more dynamic and flexible integration between the two modalities, aiming to improve classification accuracy and interpretability.

<img src="fusion_diagram.png" alt="architecture diagram" width="100%">

## Dependencies  
All required libraries are listed in the `requirements.txt` file. To install them, run:

```bash
$ pip install -r requirements.txt
```

## Usage

Once the dependencies are installed, you can train the model by running:

```bash
$ python train_attention_classifier.py
```

## Reference

If you use this code or its ideas in your research, please cite the original paper:

[Toward Next-Generation Signal Intelligence: A Hybrid Knowledge and Data-Driven Deep Learning Framework for Radio Signal Classification, IEEE, 2023.](https://ieeexplore.ieee.org/document/10042021)