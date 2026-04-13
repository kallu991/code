import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def logistic_regression(X, y, learning_rate=0.01, max_iter=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.random.randn(X_b.shape[1], 1)
    
    for i in range(max_iter):
        z = X_b.dot(theta)
        h = sigmoid(z)
        gradient = X_b.T.dot(h - y.reshape(-1, 1)) / len(X)
        theta -= learning_rate * gradient
    
    return theta

def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b.dot(theta))

def predict(X, theta):
    return (predict_proba(X, theta) >= 0.5).astype(int)

def plot_results(X, y, theta):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='o', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='s', label='Class 1')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_proba(mesh_points, theta)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression')
    plt.legend()
    plt.show()

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

theta = logistic_regression(X_train, y_train)
y_pred = predict(X_test, theta)

accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Accuracy: {accuracy:.2f}")

plot_results(X, y, theta)