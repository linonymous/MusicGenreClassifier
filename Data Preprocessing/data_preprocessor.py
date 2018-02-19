import pandas as pd
import matplotlib.pyplot as plt

def load_dataset(data_path):
    """
    This function load the dataset to X and Y variables
    :return X dataset and Y dataset

    """
    return pd.read_csv(data_path, delimiter=",")


def split_data(X, Y, data_set):
    """
    Split the dataset into training and testing dataset
    :param X: Features matrix
    :param Y: Label Matrix
    :return: returns training features, training labels, testing features, testing labels
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=333, stratify=data_set.iloc[:, -1:])
    return X_train, Y_train, X_test, Y_test


def extract_data(data_set):
    """
    Get X and Y dataset from single dataset, assumes last column in dataset is Y.
    :param data_set: dataset to be extracted
    :return: X dataset of features and Y dataset of labels
    """
    X = data_set.iloc[:, :-1]
    Y = data_set.iloc[:, -1:]
    return X, Y

if __name__ ==  "__main__" :
    data_set = load_dataset('C:/Users/Mahesh.Bhosale/PycharmProjects/MusicGenreClassifier/MusicGenreClassifier/clean_data.csv')
    X, Y = extract_data(data_set)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, data_set)
    X_train[X_train[(X_train.dtypes == "float64")|(X_train.dtypes == "int64")].index.values].hist(flgsize=[11,11])
