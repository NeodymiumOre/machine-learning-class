import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def import_data_from_csv(file):
    """Import data from given file"""

    # get data from csv files
    data = pd.read_csv("data.csv")
    data.columns = ['feature 1', 'feature 2', 'label']
    return data

def colorplot_2d_data(data, colour):
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
                marker='*',
                c=colour,
                label='0')
    plt.scatter(feature2['feature 1'].to_numpy(),
                feature2['feature 2'].to_numpy(),
                s=marker_size,
                marker='o',
                facecolors='none',
                edgecolors=colour,
                label='1')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    # plt.title("Visualized dataset from CSV")


if __name__ == "__main__":

    # PREPARE DATA SET

    # get data from data.csv file
    file = "data.csv"
    data = import_data_from_csv(file)

    # plot imported data and save the image
    colorplot_2d_data(data, 'b')
    plt.savefig('pictures/svc/oryginal_dataset.png')
    plt.show()

    # split dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(data[['feature 1', 'feature 2']].to_numpy(),
                                                        data['label'].to_numpy(),
                                                        random_state=1,
                                                        test_size=0.2)
    train_data = pd.DataFrame({'feature 1':x_train.T[0], 'feature 2':x_train.T[1], 'label':y_train})

    # PERFORM CLASSIFICATION WITH DEFAULT HYPERPARAMETERS

    svc_model = SVC()
    svc_model.fit(x_train, y_train)
    tree_predictions = svc_model.predict(x_test)
    score = svc_model.score(x_test, y_test)
    print(f"Accuracy score of SVC with default hyperparameters: {score:.4}")
    print("Default hyperparameters:")
    print(f"{svc_model.get_params()}")

    # EXPERIMENT WITH HYPERPARAMETERS

    # choose hyperparameters for test with their values
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    c = [0.01, 0.1, 1, 10, 100]
    degree = [2, 3, 4]
    gamma = [0.01, 0.1, 1, 10, 100]

    parameters = dict(kernel=kernel, C=c, degree=degree, gamma=gamma)

    # create model for grid searching
    gs_model = GridSearchCV(SVC(random_state=1), parameters, n_jobs=-1)
    gs_model.fit(data[['feature 1', 'feature 2']].to_numpy(), data['label'].to_numpy())

    # prnit best result
    print("Best hyperparameters' values from grid search:")
    print(f"{str(gs_model.best_estimator_)}, {gs_model.best_score_:.4}")

    # PLOT RESULTS

    # create subsets of data
    data_train = pd.DataFrame({'feature 1': x_train.T[0], 'feature 2': x_train.T[1], 'label': y_train})
    data_test = pd.DataFrame({'feature 1': x_test.T[0], 'feature 2': x_test.T[1], 'label': y_test})

    # plot results
    colorplot_2d_data(data_train, 'k')
    colorplot_2d_data(data_test, 'b')
    plt.savefig('pictures/svc/fitted_dataset.png')
    plt.show()

