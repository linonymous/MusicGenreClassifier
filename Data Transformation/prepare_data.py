import csv

from feature_extractor import feature_calculator
import os
import numpy
x_path = "/home/mahesh/Mahesh/MusicGenreClassifier/music_features.csv"
y_path = "/home/mahesh/Mahesh/MusicGenreClassifier/music_genres.csv"

def calculate_features(dataset_path):
    """
    Function is used to calculate features of all the audio files in dataset_path, essentially by travelling all the music genres
    :param dataset_path: Path to the root directory of dataset, it should have sun=bdirectories of music-genres like blue,hiphop with their respective audio files in it
    :return:
    """
    features = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for name in files :
            print(os.path.join(root, name))
            print(os.path.basename(root))
            feature = feature_calculator.feature_calculator(os.path.join(root,name))
            features.append(feature)
            print(type(os.path.basename(root)))
            labels.append(os.path.basename(root))
            #
            # print(type(features))
            #write_csv(features, os.path.basename(root))
    print(type(labels))
    write_csv(features, labels, x_path, y_path)

def write_csv(features, labels, x_path, y_path):
    """
    Function is used to write features and labels to csv
    ;:param features: calculated features of music file(28 to be precise)
    ;:param labels: y labels for corresponding features
    ;:param x_path: Path of the csv file where X data set would be prepared
    ;:param y_path: Path of the csv file where y data set would be prepared
    :return:
    """

    with open(x_path, "wb") as f:
        feature_writer = csv.writer(f)
        feature_writer.writerows(features)

    with open(y_path, "w") as f:
        for label in labels:
            f.write(label + "\n")

if __name__ == "__main__":
    calculate_features('/home/mahesh/Mahesh/MusicGenreClassifier/Data Transformation/Dataset/')