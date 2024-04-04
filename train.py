import os
import numpy as np
from joblib import dump, load

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Path to the folder with images
# Labels: 0 for Photograph, 1 for Drawing
IMAGE_PATH_PHOTO = 'assets_photo/'
IMAGE_PATH_ART = 'assets_art/'

# classifier name
CLASSIFIER_NAME = 'classifier.joblib'


# If the classifier is already trained, load it
# Otherwise, load the VGG16 model pre-trained on the ImageNet dataset
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


# Get asset locations from IMAGE_PATH_ART and IMAGE_PATH_PHOTO folder
def get_image_locs():
    image_locs = []
    for filename in os.listdir(IMAGE_PATH_PHOTO):
        image_locs.append(IMAGE_PATH_PHOTO + filename)
    for filename in os.listdir(IMAGE_PATH_ART):
        image_locs.append(IMAGE_PATH_ART + filename)
    return image_locs


# Create labels from get_image_locs() data
def create_labels():
    labels = []
    for i in os.listdir(IMAGE_PATH_PHOTO):
        labels.append(0)
    for i in os.listdir(IMAGE_PATH_ART):
        labels.append(1)
    return labels


# Split the data into training and test sets
def split_data(model):
    features = []
    image_locs = get_image_locs()
    labels = create_labels()

    for img_loc in image_locs:
        features.append(extract_features(model, img_loc))
    
    if (len(features) != len(labels)):
        print('Error: number of features and labels do not match')
        return None, None, None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


# Train a SVM classifier on the features
def train(X_train, y_train):
    if (not X_train or not y_train):
        print('Error: features or labels not found')
        return None

    if os.path.exists(CLASSIFIER_NAME):
        clf = load(CLASSIFIER_NAME)
    else:
        clf = SVC()
        clf.fit(X_train, y_train)
    return clf


def test_classifier(clf, X_test, y_test):
    if (not clf or not X_test or not y_test):
        print('Error: classifier not found')
        return

    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))


# Save the classifier
def save_classifier(clf):
    if (not clf):
        print('Error: classifier not found')
        return

    dump(clf, CLASSIFIER_NAME)


# main function
def main():
    # load the VGG16 model
    model = load_model()

    # split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(model)

    # train a SVM classifier on the features
    clf = train(X_train, y_train)

    # test the classifier
    test_classifier(clf, X_test, y_test)

    # save the classifier
    save_classifier(clf)


# call main function
if __name__ == "__main__":
    main()
