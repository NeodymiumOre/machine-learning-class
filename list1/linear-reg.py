import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split


def generate_data_points():
    """Generate signal with noise"""

    x = 0.4 * np.linspace(-3, 3, 500).reshape(500, 1)
    y = 6 + 4 * x + np.random.randn(500, 1)

    return x, y

def plot_2d_data_and_predictions(x, y, y_pred, title, equation):
    """Plot data points and model predictions on the same plot"""

    plt.scatter(x, y, label="Data")
    plt.plot(x, y_pred, color='black', linewidth=2, label="Predictions")
    plt.text(0.05, 0.9, equation, transform=plt.gca().transAxes, fontsize=12, va="bottom", ha="left", bbox=dict(facecolor='white', alpha=0.8))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="lower right")
    plt.show()

def get_ls_prediction(x_train, y_train, x_test):
    """Return prediction based on training set from LS model"""

    # least squares linear rregression prediction
    ls_model = LinearRegression()
    ls_model.fit(x_train, y_train)
    ls_predictions = ls_model.predict(x_test)

    # get the equation
    ls_equation = f"y = {ls_model.coef_[0][0]:.2f}x + {ls_model.intercept_[0]:.2f}"

    return ls_predictions, ls_equation

def get_sgd_prediction(x_train, y_train, x_test):
    """Return prediction based on training set from SGD model"""

    sgd_model = SGDRegressor(learning_rate='constant', eta0=lr, max_iter=iterations, random_state=0)
    sgd_model.fit(x_train, y_train)
    sgd_predictions = sgd_model.predict(x_test)

    # get the equation
    sgd_equation = f"y = {sgd_model.coef_[0]:.2f}x + {sgd_model.intercept_[0]:.2f}"

    return sgd_predictions, sgd_equation


if __name__ == "__main__":

    # PREPARE DATA SET

    # generate data set
    x, y = generate_data_points()
    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

    # PERFORM LS REGRESSION

    # train model and get predictions
    ls_predictions, ls_equation = get_ls_prediction(x_train, y_train, x)
    # plot data and predictions for least lquares
    title = "Least Squares Linear Regression"
    plot_2d_data_and_predictions(x, y, ls_predictions, title, ls_equation)

    # PERFORM SGD REGRESSION

    # create lists of values for hyperparameters
    learning_rates = [0.01, 0.1, 0.5]
    num_iterations = [100, 1000, 5000]

    for lr in learning_rates:
        for iterations in num_iterations:

            # train model and get predictions
            sgd_predictions, sgd_equation = get_sgd_prediction(x_train, y_train, x)
            
            # plot data and predictions for SGD method
            title = f"SGD Linear Regression (LR={lr}, Iterations={iterations})"
            plot_2d_data_and_predictions(x, y, sgd_predictions, title, sgd_equation)
