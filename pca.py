import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    
    cov_matrix = np.cov(X_centered.T)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    components = eigenvectors[:, :n_components]
    
    X_transformed = X_centered.dot(components)
    
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_transformed, components, explained_variance_ratio

def inverse_transform(X_transformed, components, mean):
    return X_transformed.dot(components.T) + mean

def plot_results(X_original, X_transformed, y, explained_variance_ratio):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['red', 'blue', 'green']
    for i in range(len(np.unique(y))):
        mask = y == i
        ax1.scatter(X_original[mask, 0], X_original[mask, 1], 
                   c=colors[i], alpha=0.6, label=f'Class {i}')
    
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title('Original Data (First 2 Features)')
    ax1.legend()
    
    for i in range(len(np.unique(y))):
        mask = y == i
        ax2.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                   c=colors[i], alpha=0.6, label=f'Class {i}')
    
    ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
    ax2.set_title('PCA Transformed Data')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_explained_variance(explained_variance_ratio):
    plt.figure(figsize=(10, 6))
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    
    plt.tight_layout()
    plt.show()

iris = load_iris()
X, y = iris.data, iris.target

X_transformed, components, explained_variance_ratio = pca(X, n_components=2)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
print(f"Explained variance ratio: {explained_variance_ratio}")
print(f"Total explained variance: {np.sum(explained_variance_ratio):.2%}")

plot_results(X, X_transformed, y, explained_variance_ratio)

X_transformed_full, components_full, explained_variance_full = pca(X, n_components=4)
plot_explained_variance(explained_variance_full)