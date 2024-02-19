import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def generate_data_points():
    """Generate signal with noise"""

    x = 0.1 * np.linspace(-10, 10, 500).reshape(500, 1)
    y = 3 * x**3 + 0.5 * x**2 + x + 2 + np.random.randn(500, 1)

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

def perform_polynomial_regression(x_train, y_train, x_test, deg):
    """Perform polynomial regression and plot results"""

    # add power of existing feature as a new feature
    new_feature = PolynomialFeatures(degree=deg, include_bias=False)
    x_train_poly = new_feature.fit_transform(x_train)
    x_test_poly = new_feature.fit_transform(x_test)
    
    # use LR with added new feature
    lin_reg = LinearRegression()
    lin_reg.fit(x_train_poly, y_train)
    lin_prediction = lin_reg.predict(x_test_poly)

    # get the equation
    coeff = lin_reg.coef_[0]
    intercept = lin_reg.intercept_[0]
    lin_equation = f"{intercept:.2f}"
    for i in range(0, deg):
        lin_equation = f"{coeff[i]:.2f}x^{i+1} + " + lin_equation
    lin_equation = "y = " + lin_equation

    return lin_prediction, lin_equation

def calculate_mse(x_oryginal, x_pred):
    """Calculate Mean Square Error for predicted regression line"""

    mse = mean_squared_error(x_oryginal, x_pred)

    return mse

if __name__ == "__main__":

    # PREPARE DATA SET

    # generate data set
    x, y = generate_data_points()
    # split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # PERFORM POLYNOMIAL REGRESSION

    # train polynomial regression with different degrees of polynomial
    for deg in range(2, 5):

        # train model and get predictions
        poly_prediction, poly_equation = perform_polynomial_regression(x_train, y_train, x, deg)

        # calculate and print MSE
        mse = calculate_mse(x, poly_prediction)
        print("MSE for degree={}: {:.3f}".format(deg, mse))

        # plot data and predictions
        title = f"Polynomial regression (degree={deg})"
        plot_2d_data_and_predictions(x, y, poly_prediction, title, poly_equation)
