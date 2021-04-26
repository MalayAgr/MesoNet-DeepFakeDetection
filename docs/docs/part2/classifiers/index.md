---
layout: default
title: "classifiers App"
nav_order: 2
parent: "Part 2: API"
has_toc: false
---

## <!-- omit in toc --> Table of Contents

- [Introduction](#introduction)
- [Database Models](#database-models)

## Introduction

The `classifiers` app provides most of the functionality in this API.

Specifically, it provides the following:

- A database model to easily upload trained models as HDF5 files. The model also supports uploading loss curves, calculating accuracy, obtaining classification reports, visualizing convolutional layers, etc.
- Serializers which convert complex Python datatypes to primitive ones so that it is easier to work with JSON.
- Utility functions to load datasets and obtain predictions in a nice format.

It is completely isolated from the rest of the project and so, extremely portable. For all intents and purposes, you can copy over the directory to your own Django project and use all of the functionality provided (provided you define some [settings](../settings)).

## Database Models
