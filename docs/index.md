---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Home
nav_order: 1
---

The problem of misinformation has concerned me for a long time. Having witnessed the drastic effects of it in both my country and elsewhere, I think my concerns are rightly placed.

I made a small attempt in doing something about it.

This project is part of the requirements to finish my Bachelor's degree in Computer Science (2017-2021).

It aims to demonstrate a solution to a small part of the misinformation problem. In particular, I detail here my approach in implementing a CNN-based DeepFake detector, first detailed in a paper published by Darius Afchar ([Github](https://github.com/DariusAf)) et al. in 2018 [[1]](#ref-1), called **MesoNet**. The official implementation (without any training code) is available [here](https://github.com/DariusAf/MesoNet).

The overall project consists of three parts:

- [Part 1: Model Construction and Training](https://github.com/MalayAgr/MesoNet-DeepFakeDetection) - This builds and trains various MesoNet variants, with the objective of obtaining multiple well-performing variants in the end. It is implemented using [TensorFlow](https://github.com/tensorflow/tensorflow).
- [Part 2: API](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API) - This is an API that can be used to fetch results from a trained MesoNet model. It is implemented using [Django](https://github.com/django/django) and the [Django Rest Framework](https://github.com/encode/django-rest-framework).
- [Part 3: Frontend](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-WebApp) - This is a webapp app which uses the above API to allow any Internet user to explore the inner workings of MesoNet. It is implemented in [Node.js](https://github.com/nodejs/node).

Here, you can find documentation on all three parts. More information about the architecture of MesoNet and the dataset used for training is available in the [README](https://github.com/MalayAgr/MesoNet-DeepFakeDetection#22-the-model) of Part 1.

{: .fs-1 }
<a  id="ref-1">[1]</a> Afchar, Darius, et al. [Mesonet: a compact facial video forgery detection network](https://arxiv.org/abs/1809.00888).
