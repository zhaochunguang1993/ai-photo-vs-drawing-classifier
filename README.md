# Photo vs Drawing Recognition Classifier

This project involves creating a lighthearted, AI-driven classifier that can distinguish between photographs and drawings.

## Backgrounds

The project was initially conceived to playfully tease a friend known for taking humorously poor-quality photos, often resembling drawings more than actual photographs. The aim was to automatically alter these photos in a comedic manner, by applying effects like adding fog or implementing exaggerated zooms.

However, due to the limitations of the free and basic tier of the Twitter API, which no longer supports essential features for fetching and manipulating images from Twitter, the project had to be scaled down. It now serves solely as a basic photo vs. drawing classifier. The inability to access and edit photos directly from Twitter as originally intended has led to the project being archived in this reduced capacity, focusing only on classifying images rather than transforming them in the humorous way that was initially planned.

## Prerequisites

- Python 3.x

## Installation

1. Create and activate a virtual environment:

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

2. Install the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Configuration

1. Create a Twitter Developer Account and create a new app to obtain the necessary API keys and access tokens.

2. *No longer used* Update the following variables with your Twitter API credentials:

  ```python
  CONSUMER_KEY = 'your_consumer_key'
  CONSUMER_SECRET = 'your_consumer_secret'
  ACCESS_TOKEN = 'your_access_token'
  ACCESS_TOKEN_SECRET = 'your_access_token_secret'
  ```

3. Add training images to the `assets_photo` and `assets_art` folders. The classifier expects the following folder structure:

  ```bash
  photo_1.jpg
  photo_2.jpg
  ...
  drawing_1.jpg
  drawing_2.jpg
  ...
  ```

## Usage

1. Generate classifier:

  ```bash
  python train.py
  ```

2. Test the classifier:

  ```bash
  python main.py
  ```
