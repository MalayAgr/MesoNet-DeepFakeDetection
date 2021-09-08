---
layout: default
title: "Part 2: API"
nav_order: 3
has_children: true
has_toc: false
---

## <!-- omit in toc --> Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Set Up](#set-up)

## Introduction

This part of the project provides a RESTful API to obtain predictions from MesoNet variants. The API is implemented using [Django](https://github.com/django/django) and the [Django REST Framework](https://github.com/encode/django-rest-framework), making available all the power of Python to you. This also allows the API to scale to an arbitrary level of complexity without breaking a sweat.

It provides a simple admin interface to upload your trained models and simple endpoints that allow a user of the API to obtain information such as a list of all the available models, the size of the dataset and predictions with visualization.

The API has been divided between two Django apps:

- [`classifiers`](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API/tree/main/mesonet_api/classifiers) - The `classifiers` app is the main workhorse of the API, providing the database model and the helper functions to upload models, load data, calculate things like accuracy and make predictions. This is a completely independent app and can easily be integrated into another Django project without breaking things (as long as the dependencies are met).
- [`api`](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API/tree/main/mesonet_api/api) - The `api` app contains all the endpoints that a user of the API can use. It relies on `classifiers` to do its work.

## Requirements

The project is created in Python 3.8.8. The other requirements are as follows:

- Django (3.2)
- Django REST Framework (3.12.4)
- Matplotlib (3.4.1)
- ScikitLearn (0.24.1)
- Sqlparse (0.4.1)
- Tensorflow (2.4.1)

## Set Up

To start using the project, follow the following steps. While the steps are specific to Linux, only the commands/methods change for Windows.

- Clone the repository from GitHub.

  ```shell
  git clone https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API.git
  ```

- Change working directory to `MesoNet-DeepfakeDetection-API/mesonet_api/`.

  ```shell
  cd MesoNet-DeepfakeDetection-API/mesonet_api/
  ```

- Install requirements. It is recommended that you use a virtual environment.

  ```shell
  pip install -r ../requirements.txt
  ```

- Create a `.env` file with at least the [`SECRET_KEY`](settings#secret_key) setting. To obtain a secret key, you can use this [tool](https://djecrety.ir/).

  ```shell
  export SECRET_KEY=<value>
  ```

  There are other [settings](settings) that you can modify. These are used in the `settings.py` file of your project. Also note that during deployment, variables in your `.env` file should NOT be preceded by `export`.

- Make sure there is a directory at `MesoNet-DeepfakeDetection-API/mesonet_api/<MEDIA_FOLDER>` with the same name as the [`DATA_ROOT`](settings#data_root) setting. This directory should contain your prediction data and have the following structure (assuming `DATA_ROOT` is `data`). Here, [`MEDIA_FOLDER`](settings#media_folder) refers to the folder where user-uploaded content should be stored:

  ```shell
  └── data/
      ├── real/
      │   ├── img1.png
      │   └── img2.png
      └── forged/
          ├── img1.png
          └── img2.png
  ```

  Note that `real` and `forged` can be renamed to change the class labels used in the API.

- Add the environment variables to your current environment.

  ```shell
  source .env
  ```

- Run migrations.

  ```shell
  python manage.py migrate
  ```

  While you don't need to, you can first run `python manage.py makemigrations` as a sanity check to make sure your migrations are up to date.

- Run the development server.

  ```shell
  python manage.py runserver
  ```

- Done!
