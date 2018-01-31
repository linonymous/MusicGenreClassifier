import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split


def load_dataset(data_path):
    """
    This function load the dataset to X and Y variables
    :return X dataset and Y dataset

    """
    data = np.genfromtxt(data_path, delimiter=",", dtype="|U5")
    return data


def split_data(X, Y):
    """
    Split the dataset into training and testing dataset
    :param X: Features matrix
    :param Y: Label Matrix
    :return: returns training features, training labels, testing features, testing labels
    """
    X_train, X_test = train_test_split(X, test_size=0.1)
    Y_train, Y_test = train_test_split(Y, test_size=0.1)
    return X_train, Y_train, X_test, Y_test


def extract_data(data_set):
    """
    Get X and Y dataset from single dataset, assumes last column in dataset is Y and skips first row
    being features names.
    :param data_set: dataset to be extracted
    :return: X and Y datasets
    """
    data_set = data_set[1:, :]
    np.random.shuffle(data_set)
    X, Y = data_set[:, :-1], data_set[:, -1]
    Y[Y == "blues"] = 0
    Y[Y == "class"] = 1
    Y[Y == "hipho"] = 2
    Y[Y == "regga"] = 3
    Y[Y == "jazz"] = 4
    Y[Y == "disco"] = 5
    Y[Y == "pop"] = 6
    Y[Y == "rock"] = 7
    Y[Y == "count"] = 8
    Y[Y == "metal"] = 9
    print(len(X[1]))
    return X, Y

'''
Best found for classes 1,2,3,4,5,7 --> 29.30% 
'''


def train_model(X_train, Y_train):
    """
    Train the model on ,logistic regression
    :param X_train: Training features
    :param Y_train: Training target class
    :return: returns trained model
    """
    modl = SVC(decision_function_shape='ovo')
    modl.fit(X_train, Y_train)
    return modl


if __name__ == "__main__":
    data_set = load_dataset('/home/mahesh/Mahesh/MusicGenreClassifier/clean_data.csv')
    X, Y = extract_data(data_set)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    model = train_model(X_train, Y_train)
    print(model.score(X_test, Y_test))