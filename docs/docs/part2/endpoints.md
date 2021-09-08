---
layout: default
title: "API Endpoints"
nav_order: 1
parent: "Part 2: API"
has_toc: false
---

| **Domain** | <https://www.mesonetdetect.games/> |

## <!-- omit in toc --> Table of Contents

- [Dataset Size](#dataset-size)
- [List Models](#list-models)
- [Prediction Results](#prediction-results)

---

## Dataset Size

Returns JSON data about the number of images available in the dataset on the server.

**URL**: `/api/dataset-size/`

**Django View**: `api.views.DatasetSizeView` [[source]](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API/blob/8961684fb1c56db99dd505eb14e539765ef83036/mesonet_api/api/views.py#L9)

**Method**: `GET`

**URL Parameters**: None

**Data Parameters**: None

**Success Response**:

```json
HTTP/1.1 200 OK
Server: nginx/1.18.0 (Ubuntu)
Date: Wed, 28 Apr 2021 16:33:47 GMT
Content-Type: application/json
Content-Length: 13
Connection: keep-alive
Vary: Accept, Cookie
Allow: GET, HEAD, OPTIONS
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Referrer-Policy: same-origin

{"size": 1945}
```

**Response Fields**:

- `size`: Number of images in the dataset.

---

## List Models

Returns JSON data listing all the MesoNet variants available to obtain predictions from.

**URL**: `/api/available-models/`

**Django View**: `api.views.ListModelsView` [[source]](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API/blob/8961684fb1c56db99dd505eb14e539765ef83036/mesonet_api/api/views.py#L17)

**Method**: `GET`

**URL Parameters**: None

**Data Parameters**: None

**Success Response** (showing only one item):

```json
HTTP/1.1 200 OK
Server: nginx/1.18.0 (Ubuntu)
Date: Wed, 28 Apr 2021 16:37:57 GMT
Content-Type: application/json
Content-Length: 1054
Connection: keep-alive
Vary: Accept, Cookie
Allow: GET, HEAD, OPTIONS
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
Referrer-Policy: same-origin

[
   {
      "model_id":"5aad0bb4-2d4c-4846-8636-85e0eaa0cbe3",
      "model_name":"Model 1",
      "model_desc":"Trained for 17 epochs with validation accuracy of 92.52%",
      "loss_curve":"https://www.mesonetdetect.games/media/loss_curves/model_1_curve.png",
      "accuracy":96.25,
      "clr":{
         "rows":[
             ["0", 0.96, 0.94, 0.95, 773],
             ["1", 0.96, 0.97, 0.97, 1172],
             ["accuracy", "", "", 0.96, 1945],
             ["macro avg", 0.96, 0.96, 0.96, 1945],
             ["weighted avg", 0.96, 0.96, 0.96, 1945]
         ],
         "columns":["", "precision","recall","f1-score","support"]
      },
      "conv_layers":[
         [8, [3, 3]],
         [8, [5, 5]],
         [16, [5, 5]],
         [16, [5, 5]]
      ]
   }
]
```

**Response Fields**:

Information about each variant includes the following -

- `model_id` (_string_) - Unique ID for the variant.
- `model_name` (_string_) - Unique user-friendly name for the variant.
- `model_desc` (_string_) - One-line description of the variant (training, accuracy, etc.).
- `loss_curve` (_string_) - URL to the variant's loss curve.
- `accuracy` (_float_) - Measured accuracy of the variant on the dataset available on the server.
- `clr` (_json_) - Classification report on the dataset available on the server. This has the following fields:
  - `columns` (_array_) - Column headers for the classification report.
  - `rows` (_array_ of _arrays_) - Rows of the classification report.
- `conv_layers` (_array_ of _arrays_) - Number of filters and their size for each convolutional layer.

---

## Prediction Results

Returns JSON data consisting of predictions made using the given MesoNet variant for the given number of images. Optionally, it also includes URLs to plots of specified convolutional layers for each image.

**URL**: `/api/predictions/:modelID/:numImgs/[:convIdx]/`

**Django View**: `api.views.PredictionResultsView` [[source]](https://github.com/MalayAgr/MesoNet-DeepfakeDetection-API/blob/8961684fb1c56db99dd505eb14e539765ef83036/mesonet_api/api/views.py#L23)

**Method**: `GET`

**URL Parameters**:

- `modelID` (required) - Unique ID of the variant that should be used for obtaining predictions. It should be a string representing a UUID.
- `numImgs` (required) - Number of images on which a prediction should be made. It should be an integer not greater than the number of images obtained from the [Dataset Size](#dataset-size) endpoint.
- `convIdx` (optional) - Indices of the convolutional layers (0-indexed) whose plots should be included in the prediction. It should be a string of indices. For example, if you want the plots of the first and third convolutional layer, this parameter should be `02`. Similarly, for the first, second and third layers, it should be `012`. Note that, in the response, the order of the plots will be in the same as that of the indices in this parameter.

**Data Parameters**: None

**Success Response** (showing only one item):

- With the `convIdx` parameter (showing only one image, plots are for the first and second layer):

  ```json
  HTTP/1.1 200 OK
  Server: nginx/1.18.0 (Ubuntu)
  Date: Wed, 28 Apr 2021 19:26:40 GMT
  Content-Type: application/json
  Content-Length: 204
  Connection: keep-alive
  Vary: Accept, Cookie
  Allow: GET, HEAD, OPTIONS
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: same-origin


  [
   {
      "img_url":"/media/data/real/real07167.jpg",
      "true_label":"real",
      "pred_label":"real",
      "probability":91.3399,
      "plots":[
         "/media/plots/plot_img1_conv0_SYO1pyT.png",
         "/media/plots/plot_img1_conv1_RYT0MMP.png"
      ]
   }
  ]
  ```

- Without the `convIdx` parameter (showing only one image):

  ```json
  HTTP/1.1 200 OK
  Server: nginx/1.18.0 (Ubuntu)
  Date: Wed, 28 Apr 2021 19:29:49 GMT
  Content-Type: application/json
  Content-Length: 118
  Connection: keep-alive
  Vary: Accept, Cookie
  Allow: GET, HEAD, OPTIONS
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: same-origin


  [
   {
      "img_url":"/media/data/real/real02303.jpg",
      "true_label":"real",
      "pred_label":"real",
      "probability":98.422,
      "plots":[]
   }
  ]
  ```

**Response Fields**:

Information about each variant includes the following -

- `img_url` (_string_) - Relative URL of the image file with respect to the domain.
- `true_label` (_string_) - Actual class label of the image (`real`/`deepfake`).
- `pred_label` (_string_) - Predicted class label of the image (`real`/`deepfake`).
- `probability` (_float_) - Predicted sigmoid probability of the image in percentage. Note that this probability always indicates the likelihood of image belonging to the predicted label, irrespective of whether the label represents the positive or the negative class. For example, above, the 98.422% is the probability of the image being real. If the predicted label was `deepfake`, 98.422% would be the probability of the image being a deepfake.
- `plots` (_array_ of _strings_): Relative URLs of the plots of convolutional layers for the image with respect to the domain. It is an empty array when `convIdx` is not provided.
