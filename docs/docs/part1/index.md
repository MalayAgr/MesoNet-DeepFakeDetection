---
layout: default
title: "Part 1: Model Construction and Training"
nav_order: 2
has_children: true
has_toc: false
---

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Set up](#set-up)

## Introduction

This part of the project deals with constructing, training, obtaining predictions from and visualizing different MesoNet variants.

Additionally, it provides a command line interface to do the above tasks.

The functionality has been divided between two packages.

- [`mesonet`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/tree/main/mesonet) - This is the main package containing modules which construct and build MesoNet variants. While not currently set up as a PyPI package, you can copy the directory to your project and obtain the necessary functionality to build, train and obtain predictions from MesoNet.

- [`cli`](https://github.com/MalayAgr/MesoNet-DeepFakeDetection/tree/main/cli) - This package provides a command line interface (CLI) that can be used to both train and obtain predictions from MesoNet. This allows you to use MesoNet without tinkering with the code. Currently, it only supports training the architecture as detailed in the paper but can be easily extended by adding appropriate functions to act as a custom model builder.

## Requirements

The project was developed in Python 3.8.8. The other requirements are as follows:

- Tensorflow (2.4.1)
- Matplotlib (3.4.1)
- ScikitLearn (0.24.1)
- Tabulate (0.8.9)

## Set Up

To start using the project, follow the following steps:

- Clone the repository from GitHub:

    ```shell
    git clone https://github.com/MalayAgr/MesoNet-DeepFakeDetection.git
    ```

- Change directory to `MesoNet-DeepFakeDetection`.

- Install requirements. It is recommended that you use a virtual environment.

    ```shell
    pip install -r requirements.txt
    ```

- Done!
