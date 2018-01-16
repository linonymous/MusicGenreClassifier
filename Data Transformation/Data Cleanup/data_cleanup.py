import pandas as pd
import math


def read_file(path_x, path_y):
    """
    This function reads the csv of input and output and returns a combined data frame
    :param path_x: path to the file of input features
    :param path_y: path to the file of output classes(genres)
    :return: the data frame where the last column is the output classes appended to features
    """
    df_music = pd.read_csv(path_x, names=range(0, 28))
    df_genres = pd.read_csv(path_y, names=range(28,29))
    result = pd.concat([df_music, df_genres], axis=1)
    return result


def data_cleanup(dataFrame):
    """
    This function replaces NaN and -inf values
    :param dataFrame: dataframe object
    :return: returns the dataFrame object without NaN and -inf values
    """
    col = len(dataFrame.columns)
    df = dataFrame[~dataFrame[:].isin([float("-inf")])]
    min = df.groupby([col-1]).min()
    for index, row in dataFrame.iterrows():
        for i in range(0, col-1):
            if row[i] == float("-inf"):
                genre = row[col - 1]
                row_num = [x for x in range(len(list(min.index))) if list(min.index)[x] == genre][0]
                print dataFrame.at[index, i]
                print min.iloc[row_num, i]
                dataFrame.at[index, i] = min.iloc[row_num, i]
                print "Hello"
    a = dataFrame.groupby([col-1]).mean()
    b = False
    for index, row in dataFrame.iterrows():
        for i in range(0, col-1):
            if math.isnan(float(row[i])):
                genre = row[col-1]
                row_num = [x for x in range(len(list(a.index))) if list(a.index)[x] == genre][0]
                dataFrame.at[index, i] = a.iloc[row_num, i]
    return dataFrame


if __name__ == "__main__":
    music = "C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\music_features.csv"
    genres = "C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\music_genres.csv"
    data_cleanup(read_file(music, genres))
    # a = range(0,28)
    # df_music = pd.read_csv(music, names=a)
    # df_genres = pd.read_csv(genres)
    # print df_music.head(5)
    # print df_genres.head(5)