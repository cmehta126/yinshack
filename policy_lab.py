# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 10:41:02 2017

@author: Mehraveh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:04:38 2016

@author: Mehraveh
"""

%reset
import numpy as np
from sklearn import svm
from sklearn.metrics import zero_one_loss
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor #GBM algorithm
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy
import imp
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error




#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin


import pandas as pd
offc = pd.read_csv('/Users/Mehraveh/Desktop/policy_lab/toy.officer_data.csv',sep=',')
cmpl = pd.read_csv('/Users/Mehraveh/Desktop/policy_lab/toy.complaint_data.csv', sep=',')


X = offc
X = X.drop(X.columns[[0,1,5,7,11]], axis=1)
X[X.columns[0]] = X[X.columns[0]].astype(str).str[0:4].astype(int)



X_cat = X[X.columns[[1,2]]]
Y=X_cat.apply(LabelEncoder().fit_transform)

X[X.columns[[1,2]]] = Y
np.where(np.isnan(X))
X.iat[np.where(np.isnan(X))[0][0],np.where(np.isnan(X))[1][0]]=0

X_race = pd.get_dummies(X['race'])

X_race.rename(columns={'0': 'race0', '1': 'race1', '2': 'race2', '3': 'race3'})

#Y_rank = pd.get_dummies(Y['rank'])

from sklearn import linear_model
clf = linear_model.LinearRegression()
X2=X[X.columns[[6,7,8,9]]]

clf.fit(X2,Y)
preditcs = clf.predict(X2)

A = np.where(X2)[1]
plt.scatter(Y,predicts, c=A,s=100)




predicts = cross_val_predict(clf, X2, Y, cv=2)
S=[Y,predicts]
mean_squared_error(Y, predicts)
fig = plt.figure(figsize=(10,10))






plt.hist(X['race'], bins='auto')
plt.hist(X['gender'], bins='auto')
plt.hist(X['age'], bins='auto')
plt.hist(X['appointed.date'], bins='auto')

X = X.drop(X.columns[[1]], axis=1)
X = X.join(X_race, how='outer')

Y = cmpl.groupby('officer_id').count()['crid']


cut_points = [2,3,4,5]
#
#
y = binning(Y, cut_points)
pd.Categorical(y)
y=pd.Categorical(y)
pd.CategoricalIndex(y)


le = preprocessing.LabelEncoder()
le.fit(y)
le.classes_
y=le.transform(y)

np.sum(y==0)
np.sum(y==1)
np.sum(y==2)
np.sum(y==3)
np.sum(y==4)



CV=2

clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,min_samples_split=5,
        min_samples_leaf=15,max_depth=5,max_features='sqrt',random_state=12,
        subsample =0.8,verbose=0,warm_start = True)


predicts = cross_val_predict(clf, X, y, cv=2)
print(np.mean(predicts==Y))

mean_squared_error(Y, predicts)

plt.plot(Y,predicts,'*')



plt.hist(Y, bins='auto')





offc = offc.merge(freq, left_on='officer_id', right_index=1, how='outer')
offc['n_crimes'] = freq



import matplotlib.pyplot as plt
rng = np.random.RandomState(10)  # deterministic random data

plt.hist(y, bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.hist(predicts, bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()









def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    x_min = x_min-1
    x_max= x_max+1
    X = (X - x_min) / (x_max - x_min)
    fig = pylab.figure(figsize=(8,8))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1],str(y[i]),
                 color=plt.cm.Set1(y[i]/max(y)),
                fontdict={'weight': 'bold', 'size': 20})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



pca = PCA(n_components=2).fit_transform(X)
plot_embedding(pca)
#y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(data)
y_pred = KMeans(init='k-means++', n_clusters=3, n_init=10).fit_predict(X)

#plt.subplot(121)
#plt.scatter(pca[:, 0], pca[:, 1], c=y_pred,s=100 , linewidths=0)
#plt.subplot(122)
#plt.scatter(pca[:, 0], pca[:, 1], marker='o',c=y,s=100 )


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X['race'],X['gender'],Y, c='r', marker='o')

ax.set_xlabel('Race')
ax.set_ylabel('Sex')
ax.set_zlabel('crimes')

plt.show()

fig.savefig('/Users/Mehraveh/Desktop/test1.png')



