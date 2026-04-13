import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (1 + np.dot(x1, x2)) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, kernel_func, **kwargs):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j], **kwargs)
    return K

def svm_train(X, y, C=1.0, kernel_func=linear_kernel, max_iter=1000, tol=1e-6, **kernel_kwargs):
    n = X.shape[0]
    alpha = np.zeros(n)
    b = 0
    
    K = compute_kernel_matrix(X, kernel_func, **kernel_kwargs)
    
    for iteration in range(max_iter):
        alpha_old = alpha.copy()
        
        for i in range(n):
            E_i = np.sum(alpha * y * K[i, :]) + b - y[i]
            
            if (y[i] * E_i < -tol and alpha[i] < C) or (y[i] * E_i > tol and alpha[i] > 0):
                j = np.random.choice([x for x in range(n) if x != i])
                E_j = np.sum(alpha * y * K[j, :]) + b - y[j]
                
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                
                if L == H:
                    continue
                
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue
                
                alpha[j] -= y[j] * (E_i - E_j) / eta
                alpha[j] = max(L, min(H, alpha[j]))
                
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue
                
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                
                b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
        
        if np.allclose(alpha, alpha_old, atol=tol):
            break
    
    support_vectors_idx = alpha > 1e-5
    return alpha, b, support_vectors_idx

def svm_predict(X_train, y_train, X_test, alpha, b, support_vectors_idx, kernel_func=linear_kernel, **kernel_kwargs):
    predictions = []
    
    for x_test in X_test:
        decision_value = 0
        for i in range(len(X_train)):
            if support_vectors_idx[i]:
                decision_value += alpha[i] * y_train[i] * kernel_func(X_train[i], x_test, **kernel_kwargs)
        decision_value += b
        predictions.append(1 if decision_value >= 0 else -1)
    
    return np.array(predictions)

def plot_svm_results(X, y, alpha, b, support_vectors_idx, kernel_func=linear_kernel, **kernel_kwargs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='s', label='Class -1', alpha=0.7)
    plt.scatter(X[support_vectors_idx, 0], X[support_vectors_idx, 1], 
                s=200, facecolors='none', edgecolors='black', linewidth=2, label='Support Vectors')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in mesh_points:
        decision_value = 0
        for i in range(len(X)):
            if support_vectors_idx[i]:
                decision_value += alpha[i] * y[i] * kernel_func(X[i], point, **kernel_kwargs)
        decision_value += b
        Z.append(decision_value)
    
    Z = np.array(Z).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], 
                linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(alpha)), alpha)
    plt.xlabel('Sample Index')
    plt.ylabel('Alpha Value')
    plt.title('Lagrange Multipliers')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    support_alpha = alpha[support_vectors_idx]
    plt.hist(support_alpha, bins=20, alpha=0.7)
    plt.xlabel('Alpha Value')
    plt.ylabel('Frequency')
    plt.title('Support Vector Alpha Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_kernels(X, y):
    kernels = [
        ('Linear', linear_kernel, {}),
        ('Polynomial', polynomial_kernel, {'degree': 3}),
        ('RBF', rbf_kernel, {'gamma': 0.1})
    ]
    
    plt.figure(figsize=(15, 5))
    
    for idx, (name, kernel_func, kwargs) in enumerate(kernels):
        alpha, b, support_vectors_idx = svm_train(X, y, kernel_func=kernel_func, **kwargs)
        
        plt.subplot(1, 3, idx + 1)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='Class +1', alpha=0.7)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='s', label='Class -1', alpha=0.7)
        plt.scatter(X[support_vectors_idx, 0], X[support_vectors_idx, 1], 
                    s=200, facecolors='none', edgecolors='black', linewidth=2, label='Support Vectors')
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = []
        for point in mesh_points:
            decision_value = 0
            for i in range(len(X)):
                if support_vectors_idx[i]:
                    decision_value += alpha[i] * y[i] * kernel_func(X[i], point, **kwargs)
            decision_value += b
            Z.append(decision_value)
        
        Z = np.array(Z).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], colors=['black'], linewidths=[2])
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'{name} Kernel SVM')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

X, y_binary = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, 
                                 n_clusters_per_class=1, random_state=42)
y = np.where(y_binary == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alpha, b, support_vectors_idx = svm_train(X_train, y_train, C=1.0)
y_pred = svm_predict(X_train, y_train, X_test, alpha, b, support_vectors_idx)

accuracy = np.mean(y_pred == y_test)
n_support_vectors = np.sum(support_vectors_idx)

print(f"Accuracy: {accuracy:.2f}")
print(f"Number of support vectors: {n_support_vectors}")
print(f"Bias term: {b:.4f}")

plot_svm_results(X_train, y_train, alpha, b, support_vectors_idx)

compare_kernels(X, y)