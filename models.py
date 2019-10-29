import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config as cf
from utils import stack_or_create, get_all_data
from sklearn.cluster import KMeans, MiniBatchKMeans
from copy import deepcopy


from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel


import math
import os
import numpy as np

weights = {}
biases = {}


class HD_CNN(nn.Module):
    def __init__(self, args):
        super(HD_CNN, self).__init__()
        self.args = args
        self.fines = {}
        for i in range (self.args.num_superclass):
            self.fines[i]=fine(self.args)
        self.croase = croase(self.args)
        self.share = share(self.args)
        #self.cluster = clustering(self.args)


class share(nn.Module):
    def __init__(self, args):
        super(share, self).__init__()
        self.args = args
        # Encoder layers
        self.enc_conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1_1 = nn.BatchNorm2d(64)
        self.enc_drop1 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn1_2 = nn.BatchNorm2d(64)
        self.enc_max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn2_1 = nn.BatchNorm2d(128)
        self.enc_drop2 =  nn.Dropout(p=self.args.drop_rate)
        self.enc_conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn2_2 = nn.BatchNorm2d(128)
        self.enc_max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x, *args, **kwargs):
        x = self.enc_drop1(self.relu(self.enc_bn1_1(self.enc_conv1_1(x))))
        x = self.enc_max_pool1(self.relu(self.enc_bn1_2(self.enc_conv1_2(x))))
        x = self.enc_drop2(self.relu(self.enc_bn2_1(self.enc_conv2_1(x))))
        x = self.enc_max_pool2(self.relu(self.enc_bn2_2(self.enc_conv2_2(x))))
        return x


class croase(nn.Module):
    def __init__(self, args):
        super(croase, self).__init__()
        self.args = args
        self.enc_conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_1 = nn.BatchNorm2d(256)
        self.enc_drop3 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_2 = nn.BatchNorm2d(256)
        self.enc_max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_1 = nn.BatchNorm2d(512)
        self.enc_drop4 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_2 = nn.BatchNorm2d(512)
        self.enc_max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512*2*2, 1024 )
        self.fc_drop1 = nn.Dropout(p=self.args.drop_rate)
        self.fc2 = nn.Linear(1*1*1024, 1024 )
        self.fc_drop2 = nn.Dropout(p=self.args.drop_rate)
        self.fc3 = nn.Linear(1*1*1024, self.args.num_fine_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def independent(self, x, *args, **kwargs):
        x = self.enc_drop3(self.relu(self.enc_bn3_1(self.enc_conv3_1(x))))
        x = self.enc_max_pool3(self.relu(self.enc_bn3_2(self.enc_conv3_2(x))))
        x = self.enc_drop4(self.relu(self.enc_bn4_1(self.enc_conv4_1(x))))
        x = self.enc_max_pool4(self.relu(self.enc_bn4_2(self.enc_conv4_2(x))))
        x = x.reshape(-1,512*2*2)
        coarse_predict = self.fc3(self.fc_drop2(self.fc2(self.fc_drop1(self.fc1(x)))))
        return F.softmax(coarse_predict, dim=1)



class fine(nn.Module):
    def __init__(self, args):
        super(fine, self).__init__()
        self.args = args
        # Independent Encoder layers
        self.enc_conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_1 = nn.BatchNorm2d(256)
        self.enc_drop3 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn3_2 = nn.BatchNorm2d(256)
        self.enc_max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_1 = nn.BatchNorm2d(512)
        self.enc_drop4 = nn.Dropout(p=self.args.drop_rate)
        self.enc_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_bn4_2 = nn.BatchNorm2d(512)
        self.enc_max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512*2*2, 1024 )
        self.fc_drop1 = nn.Dropout(p=self.args.drop_rate)
        self.fc2 = nn.Linear(1*1*1024, 1024 )
        self.fc_drop2 = nn.Dropout(p=self.args.drop_rate)
        self.fc3 = nn.Linear(1*1*1024, self.args.num_fine_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def independent(self, x, *args, **kwargs):
        x = self.enc_drop3(self.relu(self.enc_bn3_1(self.enc_conv3_1(x))))
        x = self.enc_max_pool3(self.relu(self.enc_bn3_2(self.enc_conv3_2(x))))
        x = self.enc_drop4(self.relu(self.enc_bn4_1(self.enc_conv4_1(x))))
        x = self.enc_max_pool4(self.relu(self.enc_bn4_2(self.enc_conv4_2(x))))
        x = x.reshape(-1,512*2*2)
        fine_class = self.fc3(self.fc_drop2(self.fc2(self.fc_drop1(self.fc1(x)))))
        return F.softmax(fine_class, dim=1)


class clustering(nn.Module):
    def __init__(self, args):
        self.Kernals=10

    def acc(self, ypred, y):
        assert len(y) > 0
        assert len(np.unique(ypred)) == len(np.unique(y))

        s = np.unique(ypred)
        t = np.unique(y)

        N = len(np.unique(ypred))
        F = np.zeros((N, N), dtype = np.int32)
        for i in range(N):
            for j in range(N):
                idx = np.logical_and(ypred == s[i], y == t[j])
                F[i][j] = np.count_nonzero(idx)

        return F

    # assign to the clusters (M-step)
    def get_assignments(self, X, centroids):
        dist = pairwise_distances(X, centroids)
        assign = np.argmin(dist,axis=1)
        return assign

    # compute the new centroids (E-step)
    def get_centroids(self, X, assignments):
        centroids = []
        for i in np.unique(assignments):
            centroids.append(X[assignments==i].mean(axis=0))
        return np.array(centroids)

    # initize the centroids
    def init_kmeans_plus_plus(self, X, K):
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

    def KMeans(self, X, centroids, n_iterations=5, axes=None):
        if axes is not None:
            axes = axes.flatten()
        for i in range(n_iterations):
            assignments = self.get_assignments(X, centroids)
            centroids = self.get_centroids(X, assignments)
        return assignments, centroids

    def spectral_clustering(self, A, K=2, gamma=10):
        # A = rbf_kernel(X, gamma=gamma)
        # A -= np.eye(A.shape[0]) # affinity
        A /=A.sum(axis=1)
        A = np.multiply(A,(np.ones((10,10))-np.identity(10)))
        D = A.sum(axis=1) # degree
        D_inv = np.diag(D**(-.5))
        L = (D_inv).dot(A).dot(D_inv) # laplacian
        s, Vh = np.linalg.eig(L)
        eigenvector = Vh.real[:,:K].copy()
        eigenvector /= ((eigenvector**2).sum(axis=1)[:,np.newaxis]**.5)
        centroids = self.init_kmeans_plus_plus(eigenvector, K)
        assignments, _ = self.KMeans(eigenvector, centroids)
        return assignments