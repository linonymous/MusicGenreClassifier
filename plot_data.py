import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas
from pandas.plotting import scatter_matrix


def visualize_scatter_plot():
    """
    Function is used to plot the scatter plot on the features
    :return:
    """
    x_data = np.genfromtxt('music_features.csv', delimiter=',')
    y_data = []
    with open('music_genres.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            y_data.append(int(row[0]))

    y_data = np.array(y_data)
    x_data1 = x_data[:, 13]
    x_data2 = x_data[:, 14]
    plt.scatter(x_data1, x_data2, c=y_data)
    plt.show()
