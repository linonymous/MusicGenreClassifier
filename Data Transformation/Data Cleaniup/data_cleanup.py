import pandas as pd
import math

# def filling_values(dataframe):
#     """
#     This function fills up NaN values by the average values
#     :param dataframe: DataFrame object in python
#     :return: return DataFrame
#     """

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


def group_by(dataFrame):
    """
    This function groups the rows according to the genres
    :param dataFrame: dataframe object
    :return: returns the groups
    """
    col = len(dataFrame.columns)
    a = dataFrame.groupby([col-1]).mean()
    b = False
    for index, row in dataFrame.iterrows():
        for i in range(0, col-1):
            if math.isnan(float(row[i])):
                genre = row[col-1]
                index = [x for x in range(len(list(a.index))) if list(a.index)[x] == genre]
                print row[i]
                print i
                row[i] = a.iloc[index, i].get(genre)
                print genre
                # print type(a.iloc[index, i])
                # print a.iloc[index, i].get("classical")
                # print index
                # print type(row[15])
                # print row[15] == float("-inf")
                b = True
                break
        if b is True:
            break
    b = False
    for index, row in dataFrame.iterrows():
        for i in range(0, col-1):
            if math.isnan(float(row[i])):
                genre = row[col-1]
                index = [x for x in range(len(list(a.index))) if list(a.index)[x] == genre]
                print row[i]
                print i
                row[i] = a.iloc[index, i].get(genre)
                print genre
                # print type(a.iloc[index, i])
                # print a.iloc[index, i].get("classical")
                # print index
                # print type(row[15])
                # print row[15] == float("-inf")
                b = True
                break
        if b is True:
            break
if __name__ == "__main__":
    music = "C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\music_features.csv"
    genres = "C:\Users\Swapnil.Walke\MusicGenreClassifier\Data Transformation\music_genres.csv"
    group_by(read_file(music, genres))
    # a = range(0,28)
    # df_music = pd.read_csv(music, names=a)
    # df_genres = pd.read_csv(genres)
    # print df_music.head(5)
    # print df_genres.head(5)