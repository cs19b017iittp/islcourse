# You can import whatever standard packages are required

import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_blobs, make_circles, load_digits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!


###### PART 1 ######

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  # write your code ...
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  # write your code ...
  X, y = make_circles(n_samples=n_points, shuffle=True,  factor=0.3, noise=0.05, random_state=0)
  return X,y

def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets
  # write your code ...
  digits = load_digits()
  X=digits.data
  y=digits.target
  return X,y

def build_kmeans(X=None,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  # write your code ...
  km = KMeans(n_clusters=k, random_state=0).fit(X)# this is the KMeans object
  return km

def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h,c,v = 0,0,0 # you need to write your code to find proper values
  h = "%.6f" % homogeneity_score(ypred_1, ypred_2)
  c = "%.6f" % completeness_score(ypred_1, ypred_2)
  v = "%.6f" % v_measure_score(ypred_1, ypred_2)
  return h,c,v
