---
layout: default
title: "Available Settings"
nav_order: 3
parent: "Part 2: API"
has_toc: false
---

## <!-- omit in toc --> Settings

A few settings are available which can be set through environment variables to control the APIs behavior.

- [SECRET_KEY](#secret_key)
- [STATIC_FOLDER](#static_folder)
- [STATIC_URL](#static_url)
- [MEDIA_FOLDER](#media_folder)
- [MEDIA_URL](#media_url)
- [MODEL_FOLDER](#model_folder)
- [DATA_ROOT](#data_root)

### SECRET_KEY

(_Required_)

**Environment variable**: `SECRET_KEY`

The secret key for the Django project. Django relies on this for many of its security features and it is recommended that you set it to a good value. A good tool for automatically generating a key is <https://djecrety.ir/>.

**Example**: `zyx2w5@-l4kceqvu&*30vx-ikb&1d=h=q3vf9$2-rb$bhfp@88`

### STATIC_FOLDER

(_Optional_)

**Environment variable**: `STATICFILES_DIR`

**Default**: `static`

Relative path with respect to the project where `collectstatic` should store the static files for use during deployment. This is used to build Django's [`STATIC_ROOT`](https://docs.djangoproject.com/en/3.2/ref/settings/#static-root) setting, by joining it with the `BASE_DIR` setting.

**Example**: `staticfiles`

### STATIC_URL

(_Optional_)

**Environment variable**: `STATIC_URL`

**Default**: `/static/`

URL to use when referring to static files stored in `STATIC_ROOT`. See Django's [`STATIC_URL`](https://docs.djangoproject.com/en/3.2/ref/settings/#static-url)

**Example**: `http://static.example.com/`

### MEDIA_FOLDER

(_Optional_)

**Environment variable**: `MEDIA_FOLDER`

**Default**: `media`

Relative path with respect to the project where user-uploaded content such as documents and images should be stored. This is used to build Django's [`MEDIA_ROOT`]() setting, by joining it with the `BASE_DIR` setting.

**Example**: `user_content`

### MEDIA_URL

(_Optional_)

**Environment variable**: `MEDIA_URL`

**Default**: `/media/`

URL to use when referring to user-uploaded content stored in `MEDIA_ROOT`. See Django's [`MEDIA_URL`]().

**Example**: `http://media.example.com/`

### MODEL_FOLDER

(_Optional_)

**Environment variable**: `MODEL_FILE_ROOT`

**Default**: `ml_models`

Relative path with respect to the project where the HDF5 of the uploaded models should be stored. This exists because it is better if the model files are stored separate from other user-uploaded content such as images.

**Example**: `mesonet_variants`

### DATA_ROOT

(_Optional_)

**Environment variable**: `DATA_ROOT`

**Default**: `data`

Relative path with respect to `MEDIA_ROOT` where the data to be used for prediction is stored. This is used to refer to the data folder easily inside the project. Specifically, this is used to build a absolute path to the directory, stored in `DATA_DIRECTORY`. Inside your project, you can refer to the directory as:

```python
from django.conf import settings

data = settings.DATA_DIRECTORY
```

**Example**: `mesonet_variants`
