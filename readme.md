# <!-- omit in toc --> MesoNet - A Deepfake Detector Built Using Python and Deep Learning

The problem of misinformation has concerned me for a long time. Having witnessed the drastic effects of it in both my country and elsewhere, I think my concerns are rightly placed.

Here, I make my small attempt in doing something about it.

## <!-- omit in toc --> Table of Contents

- [1. Introduction](#1-introduction)
- [2. General Approach](#2-general-approach)
- [3. References](#3-references)

## 1. Introduction

This project is part of the requirements to finish my Bachelor's degree in Computer Science (2017-2021).

It aims to demonstrate a solution to a small part of the misinformation problem. In particular, I detail here my approach in implementing a CNN-based DeepFake detector, first detailed in a paper published by Darius Afchar ([Github](https://github.com/DariusAf)) et al. in 2018 [[1]](#ref-1), called **MesoNet**.

The overall project consists of three parts:

- [Part 1: Model Construction and Training](https://github.com/MalayAgarwal-Lee/MesoNet-DeepFakeDetection) - This builds and trains various MesoNet variants, with the objective of obtaining multiple well-performing variants in the end. It is implemented using TensorFlow.
- [Part 2: API](https://github.com/MalayAgarwal-Lee/MesoNet-DeepfakeDetection-API) - This is an API that can be used to fetch results from a trained MesoNet model. It is implemented using Django and the Django Rest Framework.
- Part 3: Frontend - This is a Node.js app which uses the above API to allow any Internet user to explore the inner workings of MesoNet.

You're currently reading about Part 1.

## 2. General Approach

## 3. References

- <a  id="ref-1">[1]</a> Afchar, Darius, et al. [Mesonet: a compact facial video forgery detection network](https://arxiv.org/abs/1809.00888).
