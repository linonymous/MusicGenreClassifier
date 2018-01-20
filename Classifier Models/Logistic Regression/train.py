from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split

def load_dataset(data_path):
    """
    This function load the dataset to X and Y variables
    :return X dataset and Y dataset

    """
    data = np.genfromtxt(data_path, delimiter=",", dtype="|U5")
    return data


def split_data(X, Y):

    X_train, X_test = train_test_split(X, test_size=0.3)
    Y_train, Y_test = train_test_split(Y, test_size=0.3)
    return X_train, Y_train, X_test, Y_test


#def validate_model(X_cv, Y_cv):
#    return accuracy

def train_model(X_train, Y_train):
    logit_model = sm.Logit(Y_train, X_train)
    result = logit_model.fit()
    print(result.summary())
    #return accuracy

#def train_model_own(X_train, Y_train):
#    return accuracy

def test_model(X_test, Y_test):
    return accuracy

#def cost():
#    return cost

#def gradient(cost_function):
#    return gradient


if __name__ == "__main__":

  data_set = load_dataset('/home/mahesh/Mahesh/MusicGenreClassifier/Data Transformation/Data Cleanup/clean_data.csv')

  #Skip the header
  data_set = data_set[1:, :]
  np.random.shuffle(data_set)
  X, Y= data_set[:, :-1], data_set[:, -1]
  print(type(Y))
  X_train, Y_train, X_test, Y_test = split_data(X, Y)
  #train_model(X_train, Y_train)
  #print("Trainig Accuracy is "+ train_accuracy)
  #cv_accuracy = validate_model(X_cv, Y_cv)
  #print("Cross Validation Accuracy is " + cv_accuracy)
  #test_accuracy = test_model(X_test, Y_test)
  #print("Testing Accuracy is " + test_accuracy)
