import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel





# assign to the clusters (M-step)
def get_assignments(X, centroids):
    dist = pairwise_distances(X, centroids)
    assign = np.argmin(dist,axis=1)
    return assign

# compute the new centroids (E-step)
def get_centroids(X, assignments):
    centroids = []
    for i in np.unique(assignments):
        centroids.append(X[assignments==i].mean(axis=0))
    return np.array(centroids)

# initize the centroids
def init_kmeans_plus_plus(X, K):
    '''Choose the next centroids with a prior of distance.'''
    assert K>=2, "So you want to make 1 cluster?"
    compute_distance = lambda X, c: pairwise_distances(X, c).min(axis=1)
    # get the first centroid
    centroids = [X[np.random.choice(range(X.shape[0])),:]]
    # choice next
    for _ in range(K-1):
        proba = compute_distance(X,centroids)**2
        proba /= proba.sum()
        centroids.append(X[np.random.choice(range(X.shape[0]), p=proba)])
    return np.array(centroids)

def KMeans(X, centroids, n_iterations=5, axes=None):
    if axes is not None:
        axes = axes.flatten()
    for i in range(n_iterations):
        assignments = get_assignments(X, centroids)
        centroids = get_centroids(X, assignments)
    return assignments, centroids

def spectral_clustering(X, K=2, gamma=10):
    A = rbf_kernel(X, gamma=gamma)
    A -= np.eye(A.shape[0]) # affinity
    D = A.sum(axis=1) # degree
    D_inv = np.diag(D**(-.5))
    L = (D_inv).dot(A).dot(D_inv) # laplacian
    s, Vh = np.linalg.eig(L)
    eigenvector = Vh.real[:,:K].copy()
    eigenvector /= ((eigenvector**2).sum(axis=1)[:,np.newaxis]**.5)
    centroids = init_kmeans_plus_plus(eigenvector, K)
    assignments, _ = KMeans(eigenvector, centroids)
    return assignments