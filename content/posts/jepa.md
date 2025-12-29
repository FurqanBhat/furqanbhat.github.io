---
title: "From Reconstruction to Prediction: Joint Embedding Predictive Architectures"
date: 2025-01-01
draft: false
---

Generative Architectures have achieved great success in creating high-fidelity data, however generation is often not the end goal. Most of the time our objective is representation: obtaining high-level abstractions of data that facilitate downstream tasks like classification or planning. We need an encoder that understands the semantic structure of the data, separating the signal (optimal features) from the noise (irrelevant details).

## The Limits of Reconstruction

Historically, we have relied on Autoencoders (and more recently, Masked Autoencoders) to learn these latent representations. These models rely on a reconstruction loss which forces the network to reproduce the raw input from a compressed bottleneck (latent representation).
While effective, this approach has a fundamental misalignment. Reconstruction biases the model toward encoding high-frequency details required to render the output (like the texture of grass or static noise), rather than solely the semantic concepts. Using pixel-level reconstruction to learn high-level semantics is computationally inefficient; the model wastes capacity memorizing texture rather than learning structure.

## The Shift to Prediction

If our goal is understanding the data distribution, optimizing for pixel-perfect reconstruction is suboptimal. This is the core thesis of Joint Embedding Predictive Architectures (JEPA).

In JEPA, we abandon pixel generation. Instead, we pass context and target blocks through an encoder to get their representations. We then pass the context representation through a predictor to predict the target representation. The loss is propagated through both the predictor and the context encoder.  
Crucially, the encoder is penalized only when it fails to capture features necessary for prediction. This forces the encoder to prioritize features that are predictable (semantics) while ignoring features that are unpredictable (noise), resulting in a far richer representation. In short, the encoder gets penalized for not choosing good enough features to generate good enough representations.

## The Collapse and Asymmetric Architecture

Training in this way using one encoder and one predictor leads to model collapse, as the model learns a trivial solution: giving out a constant representation for each input. If the encoder gives out constant representation for each input, the prediction is effortless, and the loss drops to zero without the model learning anything about the data.

To prevent this collapse, JEPA uses an Asymmetric Student-Teacher Architecture.

Instead of passing context and target data through the same encoder, we use different encoders (different instances of the same encoder): a Context/Student encoder for context encoding and a Target/Teacher encoder for target encoding. The student encoder is trained as mentioned above, while the teacher encoder weights are frozen and only updated via EMA (Exponential Moving Average).


W_t = λ · W_t + (1 − λ) · W_s



This EMA leads to a lag (or friction) between the student and teacher encoders. If the student encoder tries to output a trivial solution (constant vector), the teacher encoder will be behind and still give a complex vector as output. This mismatch will result in a huge loss and a penalty for the student encoder, forcing the student encoder to change its weights and learn meaningful representations.

