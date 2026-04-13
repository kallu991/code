import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def perceptron_train(X, y, learning_rate=0.1, max_iter=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)
    
    X_with_bias = np.c_[np.ones(n_samples), X]
    
    for epoch in range(max_iter):
        errors = 0
        for i in range(n_samples):
            prediction = perceptron_predict_single(X_with_bias[i], weights)
            error = y[i] - prediction
            
            if error != 0:
                weights += learning_rate * error * X_with_bias[i]
                errors += 1
        
        if errors == 0:
            break
    
    return weights

def perceptron_predict_single(x, weights):
    return 1 if np.dot(x, weights) >= 0 else 0

def perceptron_predict(X, weights):
    n_samples = X.shape[0]
    X_with_bias = np.c_[np.ones(n_samples), X]
    predictions = []
    
    for i in range(n_samples):
        pred = perceptron_predict_single(X_with_bias[i], weights)
        predictions.append(pred)
    
    return np.array(predictions)

def plot_results(X, y, weights):
    plt.figure(figsize=(10, 6))
    
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], c='red', marker='o', label='Class 0', alpha=0.7)
    plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', marker='s', label='Class 1', alpha=0.7)
    
    if X.shape[1] == 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        if weights[2] != 0:
            x_boundary = np.linspace(x_min, x_max, 100)
            y_boundary = -(weights[0] + weights[1] * x_boundary) / weights[2]
            plt.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_training_process(X, y, learning_rate=0.1, max_iter=100):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features + 1)
    X_with_bias = np.c_[np.ones(n_samples), X]
    
    errors_per_epoch = []
    
    for epoch in range(max_iter):
        errors = 0
        for i in range(n_samples):
            prediction = perceptron_predict_single(X_with_bias[i], weights)
            error = y[i] - prediction
            
            if error != 0:
                weights += learning_rate * error * X_with_bias[i]
                errors += 1
        
        errors_per_epoch.append(errors)
        
        if errors == 0:
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors_per_epoch)), errors_per_epoch, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.title('Perceptron Training Progress')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return weights

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, 
                          n_clusters_per_class=1, random_state=42)

weights = perceptron_train(X, y)
predictions = perceptron_predict(X, weights)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")
print(f"Final weights: {weights}")

plot_results(X, y, weights)
plot_training_process(X, y)