import numpy as np
import matplotlib.pyplot as plt

def hebbian_learning(X, learning_rate=0.01, max_iter=1000):
    n_samples, n_features = X.shape
    weights = np.random.normal(0, 0.1, n_features)
    
    weight_history = [weights.copy()]
    
    for epoch in range(max_iter):
        for i in range(n_samples):
            x = X[i]
            y = np.dot(weights, x)
            
            weights += learning_rate * y * x
            
            weights = weights / np.linalg.norm(weights)
        
        weight_history.append(weights.copy())
    
    return weights, np.array(weight_history)

def oja_rule(X, learning_rate=0.01, max_iter=1000):
    n_samples, n_features = X.shape
    weights = np.random.normal(0, 0.1, n_features)
    
    weight_history = [weights.copy()]
    
    for epoch in range(max_iter):
        for i in range(n_samples):
            x = X[i]
            y = np.dot(weights, x)
            
            weights += learning_rate * y * (x - y * weights)
        
        weight_history.append(weights.copy())
    
    return weights, np.array(weight_history)

def generate_correlated_data(n_samples=200):
    np.random.seed(42)
    
    angle = np.pi / 6
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    
    data = np.random.randn(n_samples, 2)
    data[:, 0] *= 3
    data[:, 1] *= 1
    
    data = data @ rotation_matrix.T
    
    return data

def plot_hebbian_results(X, weights, weight_history, title):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    
    scale = 3
    plt.arrow(0, 0, weights[0] * scale, weights[1] * scale, 
              head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} - Final Weight Vector')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 3, 2)
    plt.plot(weight_history[:, 0], label='Weight 1')
    plt.plot(weight_history[:, 1], label='Weight 2')
    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title('Weight Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(weight_history[:, 0], weight_history[:, 1], 'b-', alpha=0.7)
    plt.scatter(weight_history[0, 0], weight_history[0, 1], c='green', s=100, label='Start')
    plt.scatter(weight_history[-1, 0], weight_history[-1, 1], c='red', s=100, label='End')
    plt.xlabel('Weight 1')
    plt.ylabel('Weight 2')
    plt.title('Weight Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def compare_learning_rules(X):
    hebbian_weights, hebbian_history = hebbian_learning(X)
    oja_weights, oja_history = oja_rule(X)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Data')
    
    scale = 3
    plt.arrow(0, 0, hebbian_weights[0] * scale, hebbian_weights[1] * scale, 
              head_width=0.2, head_length=0.3, fc='red', ec='red', linewidth=3, label='Hebbian')
    plt.arrow(0, 0, oja_weights[0] * scale, oja_weights[1] * scale, 
              head_width=0.2, head_length=0.3, fc='blue', ec='blue', linewidth=3, label='Oja')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Comparison of Learning Rules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    hebbian_norms = np.linalg.norm(hebbian_history, axis=1)
    oja_norms = np.linalg.norm(oja_history, axis=1)
    
    plt.plot(hebbian_norms, label='Hebbian Rule', linewidth=2)
    plt.plot(oja_norms, label='Oja Rule', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Weight Vector Norm')
    plt.title('Weight Vector Norms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

X = generate_correlated_data()

X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

hebbian_weights, hebbian_history = hebbian_learning(X_normalized)
print(f"Hebbian final weights: {hebbian_weights}")

oja_weights, oja_history = oja_rule(X_normalized)
print(f"Oja final weights: {oja_weights}")

plot_hebbian_results(X_normalized, hebbian_weights, hebbian_history, "Hebbian Learning")
plot_hebbian_results(X_normalized, oja_weights, oja_history, "Oja's Rule")

compare_learning_rules(X_normalized)