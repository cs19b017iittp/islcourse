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

###### PART 2 ######
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_lr_model(X=None, y=None):
  pass
  lr_model = None
  # write your code...
  # Build logistic regression, refer to sklearn
  lr_model = LogisticRegression(solver="liblinear",fit_intercept=False)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  pass
  rf_model = None
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  rf_model = RandomForestClassifier(random_state=400)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  pass
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  y_pred = model1.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average='micro')
  rec =  recall_score(y, y_pred , average='micro')
  f1 =  f1_score(y, y_pred, average='micro' )
  auc = roc_auc_score(y, model1.predict_proba(X), multi_class='ovr' )
  return acc, prec, rec, f1, auc
