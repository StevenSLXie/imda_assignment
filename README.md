# Identify the unseen Captchas 

## Overview 
- Given the nature of the CAPTCHA (created strictly following the same pattern), no advanced AI algorithm is needed for this exercise
- The basic idea is to use pattern matching. Given a new CAPTCHA, we segment it into 5 characters. Then for each character, we compare its similarity with all available characters and see which one is the best match 

## Get started 

### Run both training and prediction 
```Python
instance = Captcha('full')
instance('data/input/inputXX.jpg', 'data/result/result.txt')
```

### Run prediction only 
```Python
instance = Captcha('predict')
instance('data/input/inputXX.jpg', 'data/result/result.txt')
```

## Basic idea 

### Training Phase

- Load a set of CAPTCHA images and their corresponding text labels, convert the images to binary images
- Infer the margins of the CAPTCHA images by analyzing the fused image to determine non-character areas.
- Remove the margins and segment each CAPTCHA image into individual characters.
- Save these segmented character images as templates for later use in prediction.

### Prediction Phase

- Load the previously saved character templates and the calculated margins
- For a new CAPTCHA image, convert to binary, remove the margins and segment it into individual characters.
- Match each segmented character against the templates using a simple matching algorithm (mean squared error) to recognize the character.
- Construct the predicted CAPTCHA text from the recognized characters.

## Key Methods

- `__call__`: Handles the prediction process by calling the predict method.

- `_load_all_images`: Loads all training images and their labels.

- `_infer_margin`: Determines the margins for cropping based on fused images.

- `_read_image`: Reads and binarizes an image.

- `_remove_margins_and_segment`: Removes margins and segments the image into characters.

- `_load_templates`: Loads character templates from saved images.

- `_decode_image`: Decodes a new CAPTCHA image by matching segmented characters to the templates.

- `_match_template`: Finds the best matching character from the templates using mean squared error.



