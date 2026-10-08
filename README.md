# Hybrid Vision-Language Captioning Models Using Vision Transformers

## Overview

This repository contains the source code, training scripts, preprocessing
procedures, evaluation pipeline, and experimental configurations associated
with the study:

**Efficient ViT–GPT-2 Image Captioning under Resource-Constrained Settings:
A Comparative Study of Lightweight Encoder–Decoder Configurations**

The study evaluates two lightweight ViT–GPT-2 image-captioning configurations
under limited computational resources.

The objective is not to introduce a new architecture, but to investigate the
empirical behavior of existing Vision Transformer and GPT-2 encoder–decoder
configurations under different training constraints.

The two evaluated configurations are:

1. **ViT–GPT-2 Small**
   - pretrained ViT encoder,
   - frozen visual encoder,
   - GPT-2 Small decoder,
   - greedy decoding,
   - larger sampled training subset.

2. **Vision-GPT2**
   - pretrained ViT encoder,
   - trainable visual encoder,
   - GPT-2 Small decoder,
   - beam-search decoding,
   - smaller sampled training subset.

For clarity, **Vision-GPT2** is used only as the name of the second experimental
configuration and does not represent a newly introduced model architecture.

---

## Repository Structure

The repository is organized into the following main directories:

```text
Hybrid-vision-language-captioning-models-using-vision-transformers/
│
├── Image Captioning ViT with GPT-2/
│   ├── training scripts
│   ├── preprocessing scripts
│   ├── evaluation scripts
│   └── configuration files
│
├── VisionGPT2/
│   ├── training scripts
│   ├── preprocessing scripts
│   ├── evaluation scripts
│   └── model implementation
│
└── README.md

## Model Checkpoints

The trained model checkpoints are not stored directly in this GitHub repository because of their large file size. The experiments can be reproduced using the provided training scripts and configuration files together with the publicly available pretrained models:
- "google/vit-base-patch16-224-in21k"
- "gpt2"
When available, archived trained checkpoints are provided through the project's external archival repository.
