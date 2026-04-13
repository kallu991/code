import numpy as np
import matplotlib.pyplot as plt

def mcculloch_pitts_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

def train_mcculloch_pitts(X, y, max_iter=100):
    n_features = X.shape[1]
    weights = np.random.uniform(-1, 1, n_features)
    threshold = 0.5
    
    for epoch in range(max_iter):
        errors = 0
        for i in range(len(X)):
            prediction = mcculloch_pitts_neuron(X[i], weights, threshold)
            error = y[i] - prediction
            if error != 0:
                weights += error * X[i]
                errors += 1
        
        if errors == 0:
            break
    
    return weights, threshold

def predict_mcculloch_pitts(X, weights, threshold):
    predictions = []
    for i in range(len(X)):
        pred = mcculloch_pitts_neuron(X[i], weights, threshold)
        predictions.append(pred)
    return np.array(predictions)

def plot_results(X, y, weights, threshold):
    plt.figure(figsize=(10, 6))
    
    for i in range(len(X)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c='blue', marker='o', s=100)
        else:
            plt.scatter(X[i, 0], X[i, 1], c='red', marker='x', s=100)
    
    if X.shape[1] == 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        if weights[1] != 0:
            x_boundary = np.linspace(x_min, x_max, 100)
            y_boundary = (threshold - weights[0] * x_boundary) / weights[1]
            plt.plot(x_boundary, y_boundary, 'k--', linewidth=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('McCulloch-Pitts Neuron')
    plt.grid(True)
    plt.show()

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

weights, threshold = train_mcculloch_pitts(X, y)
predictions = predict_mcculloch_pitts(X, weights, threshold)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")
print(f"Weights: {weights}")
print(f"Threshold: {threshold}")

plot_results(X, y, weights, threshold)