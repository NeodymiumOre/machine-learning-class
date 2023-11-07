import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def import_data_from_csv(file):
    """Import data from given file"""

    # get data from csv files
    data = pd.read_csv("data.csv")
    data.columns = ['feature 1', 'feature 2', 'label']
    return data

def colorplot_2d_data(data):
    """Plot array of data on 2D colorplot"""

    # divide data into 2 groups based on label value
    feature1 = data.loc[data['label'] == 0]
    feature2 = data.loc[data['label'] == 1]

    # Plot the data set loaded from CSV.
    marker_size = 20
    plt.style.use('seaborn-v0_8')
    plt.scatter(feature1['feature 1'].to_numpy(),
                feature1['feature 2'].to_numpy(),
                s=marker_size,
                marker='o',
                c='b',
                label='0')
    plt.scatter(feature2['feature 1'].to_numpy(),
                feature2['feature 2'].to_numpy(),
                s=marker_size,
                marker='o',
                facecolors='none',
                edgecolors='k',
                label='1')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.title("Visualized dataset from CSV")
    plt.savefig("pictures/tree/xd.png")

    plt.show()


if __name__ == "__main__":

    # PREPARE DATA SET

    # get data from data.csv file
    file = "data.csv"
    data = import_data_from_csv(file)

    colorplot_2d_data(data)
