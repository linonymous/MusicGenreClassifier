import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas
from pandas.plotting import scatter_matrix


def visualize_scatter_plot(x_file, y_file, feature1, feature2):
    """
    Feature is used to plot the scatter plot of the data
    :param x_file: path to X feature csv file
    :param y_file: path to X feature csv file
    :param feature1: column number targeted to plot
    :param feature1: column number targeted to plot
    :return:
    """
    x_data = np.genfromtxt(x_file, delimiter=',')
    y_data = []
    with open(y_file, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            y_data.append(int(row[0]))

    y_data = np.array(y_data)
    x_data1 = x_data[:, feature1]
    x_data2 = x_data[:, feature2]
    plt.scatter(x_data1, x_data2, c=y_data)
    plt.show()

if __name__ == "__main__":
    visualize_scatter_plot('music_features.csv','music_genres.csv',13,14)