import os
import random
import sys
import numpy as np

import tweepy
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from joblib import dump, load


# Credentials to access Twitter API
CONSUMER_API_KEY='****'
CONSUMER_API_KEY_SECRET='****'
BEARER_TOKEN='****'
ACCESS_TOKEN='****'
ACCESS_TOKEN_SECRET='****'

CLASSIFIER_NAME = 'classifier.joblib'

# Path to the folder with images
# Labels: 0 for Photograph, 1 for Drawing
IMAGE_PATH_TEST = 'assets_test/'
IMAGE_PATH_TWEET = 'tweet_image.jpg'

# Create an OAuthHandler instance
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    consumer_key=CONSUMER_API_KEY,
    consumer_secret=CONSUMER_API_KEY_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    wait_on_rate_limit=False,
)


# Get the user id using user's screen name
def get_user_id(screen_name):
    user = client.get_user(username=screen_name)
    return user.id


# Create a stream
def create_stream():
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=client.auth, listener=myStreamListener)
    myStream.filter(track=['python'], is_async=True)
    yStream.filter(follow=["2211149702"])
    

# Create a tweet
def tweet_random_number():
    random_number = random.randint(1, 100)
    client.create_tweet(text=str(random_number))


# Load the model
def load_model():
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return model


# Extract features from an image using the VGG16 pre-trained model
def extract_features(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return np.array(features).flatten()


# Extract features from an image using the VGG16 pre-trained model
def classify_image(image_path, model, classifier):
    # Extract features
    features = extract_features(model, image_path)
    
    # Reshape features
    features = np.reshape(features, (1, -1))
    
    # Predict
    prediction = classifier.predict(features)
    
    # Return the prediction
    return prediction[0] == 0


# Get asset locations from IMAGE_PATH_TEST folder
def get_image_locs():
    image_locs = []
    for filename in os.listdir(IMAGE_PATH_TEST):
        image_locs.append(IMAGE_PATH_TEST + filename)
    return image_locs


# Main function
def main():
    # load the model and classifier
    model = load_model()
    classifier = load(CLASSIFIER_NAME)
    
    # classify the image in get_image_locs
    for image_path in get_image_locs():
        is_photo = classify_image(image_path, model, classifier)
        print(image_path, ':', is_photo)


# Call main function
if __name__ == "__main__":
    main()