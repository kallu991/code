import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def gaussian_pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma)

def em_algorithm(X, k, max_iter=100, tol=1e-6):
    n, d = X.shape
    
    mu = np.random.randn(k, d)
    sigma = np.ones(k)
    pi = np.ones(k) / k
    
    log_likelihood_old = -np.inf
    
    for iteration in range(max_iter):
        gamma = np.zeros((n, k))
        
        for i in range(n):
            for j in range(k):
                gamma[i, j] = pi[j] * gaussian_pdf(X[i], mu[j], sigma[j])
        
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        N_k = gamma.sum(axis=0)
        
        for j in range(k):
            mu[j] = (gamma[:, j].reshape(-1, 1) * X).sum(axis=0) / N_k[j]
            sigma[j] = np.sum(gamma[:, j] * np.sum((X - mu[j]) ** 2, axis=1)) / N_k[j]
            pi[j] = N_k[j] / n
        
        log_likelihood = 0
        for i in range(n):
            likelihood = 0
            for j in range(k):
                likelihood += pi[j] * gaussian_pdf(X[i], mu[j], sigma[j])
            log_likelihood += np.log(likelihood)
        
        if abs(log_likelihood - log_likelihood_old) < tol:
            break
        
        log_likelihood_old = log_likelihood
    
    return mu, sigma, pi, gamma

def predict_cluster(X, mu, sigma, pi):
    n, k = X.shape[0], len(mu)
    gamma = np.zeros((n, k))
    
    for i in range(n):
        for j in range(k):
            gamma[i, j] = pi[j] * gaussian_pdf(X[i], mu[j], sigma[j])
    
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    return np.argmax(gamma, axis=1)

def plot_results(X, labels, mu):
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i % len(colors)], alpha=0.6, label=f'Cluster {i}')
    
    for i, center in enumerate(mu):
        plt.scatter(center[0], center[1], c='black', marker='x', s=200, linewidths=3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('EM Algorithm Clustering')
    plt.legend()
    plt.show()

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.5)

k = 3
mu, sigma, pi, gamma = em_algorithm(X, k)
labels = predict_cluster(X, mu, sigma, pi)

print(f"Cluster centers: {mu}")
print(f"Cluster weights: {pi}")

plot_results(X, labels, mu)