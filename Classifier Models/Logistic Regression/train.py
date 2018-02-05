import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import  pandas as pd
import sklearn.metrics as sm
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import learning_curve


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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=333, stratify=data_set.iloc[:, -1])
    return X_train, Y_train, X_test, Y_test


def extract_data(data_set):
    """
    Get X and Y dataset from single dataset, assumes last column in dataset is Y and skips first row
    being features names.
    :param data_set: dataset to be extracted
    :return: X and Y datasets
    """
    return data_set.iloc[:, :-1], data_set.iloc[:, -1:]


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
    modl = LogisticRegression(class_weight='balanced', multi_class='ovr', penalty='l2')
    modl.fit(X_train, Y_train)
    return modl


def test_model(model, X_train, Y_train, X_test, Y_test):
    """
    Used to test the model and return testing score
    :param X_test: Testing features
    :param Y_test: Testing labels
    :param model: Model to test
    :return: returns accuracy score in %
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = sm.f1_score(Y_train, train_pred, average='weighted')
    test_score = sm.f1_score(Y_test, test_pred, average='weighted')
    return train_score, test_score


def print_training_curves(model, X, Y):
    """
    Prints training curves in score vs number of traning examples
    :param model: Trained model
    :param X: Training features
    :param Y: Training labels
    :return:
    """
    plt.figure()
    plt.title("learning_curves")
    plt.xlabel("training examples")
    plt.ylabel("Scores")
    train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X, y=Y,shuffle=True, n_jobs=100, cv=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="cross validation score")
    plt.legend(loc='best')
    plt.show()


def predict(model, X_test):
    """
    Retuns prediction on features X_test with model
    :param model:Trained model used to predict
    :param X_test:Testing features
    :return: Returns the predictions
    """
    return model.predict(X_test)


def print_confusion_matrix(model, X_test, Y_test):
    """
    Prints the confusion matrix based on testing examples
    :param model: Trained model
    :param X_test: Test features
    :param Y_test: Test labels
    :return: returns the confusion matrix
    """
    b = []
    for x in X_test:
        l = []
        for i in x:
            l.append(float(i))
        b.append(l)
    X_test = b
    predictions = predict(model, X_test)
    cm = metrics.confusion_matrix(Y_test, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=False, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    sns.heatmap(cm, cmap='Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.plot()
    plt.show()
    return cm


if __name__ == "__main__":
    data_set = load_dataset('C:/Users/Mahesh.Bhosale/PycharmProjects/MusicGenreClassifier/MusicGenreClassifier/clean_data.csv')
    X, Y = extract_data(data_set)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, data_set)
    model = train_model(X_train, Y_train)
    print(test_model(model, X_train, Y_train, X_test, Y_test))
    print_training_curves(model, X, Y)
