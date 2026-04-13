import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)

def plot_results(X, y, theta):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_plot = predict(X_plot, theta)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.show()

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

theta = linear_regression(X_train, y_train)
y_pred = predict(X_test, theta)

mse = np.mean((y_test - y_pred) ** 2)
print(f"MSE: {mse:.2f}")

plot_results(X, y, theta)