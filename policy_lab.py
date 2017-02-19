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
from sklearn import linear_model



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
X_race.columns = ['race0','race1','race2','race3']

X = X.drop(X.columns[[1]], axis=1)
X = X.join(X_race, how='outer')


crid = cmpl.groupby('officer_id').count()
dd = offc.merge(crid, left_on = 'officer_id', right_index = 1, how = 'outer')
Y=dd['crid']



clf = GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,min_samples_split=5,
        min_samples_leaf=15,max_depth=5,max_features='sqrt',random_state=12,
        subsample =0.8,verbose=0,warm_start = True)


predicts = cross_val_predict(clf, X, y, cv=2)
print(np.mean(predicts==Y))


clf = GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,min_samples_split=5,
        min_samples_leaf=15,max_depth=5,max_features='sqrt',random_state=12,
        subsample =0.8,verbose=0,warm_start = True)

predicts = cross_val_predict(clf, X, Y, cv=3)

clf.fit(X2,Y)
predicts = clf.predict(X2)

mean_squared_error(Y, predicts)
plt.plot(Y,predicts,'*')
correlation = np.corrcoef(Y, predicts)[0,1]
print(correlation)






######





import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

%matplotlib inline
python_dir = '/Users/Mehraveh/Desktop/yins-datahack-2017'
os.chdir(python_dir)

from chris_rasmus_graph_features import *
from count_lagged_complaints import *

from sklearn.preprocessing import normalize

r = add_complaints_by_year(2012, 4, cmpl, offc)


#Y = add_complaints_by_year(2015, 3, cmpl, offc)
#
#Y=Y[[Y.columns[Y.shape[1]-1],Y.columns[Y.shape[1]-2],Y.columns[Y.shape[1]-3]]]
#Y=Y.sum(axis=1)


#X = r
#X = X.drop(X.columns[[0,1,5,7,11]], axis=1)
#X[X.columns[0]] = X[X.columns[0]].astype(str).str[0:4].astype(int)
#
X_cat = X[X.columns[[1,2]]]
#XX=X_cat.apply(LabelEncoder().fit_transform)
#
#X[X.columns[[1,2]]] = XX
#np.where(np.isnan(X))
#
X_race = pd.get_dummies(X['race'])
X_race.columns = ['race0','race1','race2','race3']
#
#X = X.drop(X.columns[[1]], axis=1)
#X = X.join(X_race, how='outer')


    


crid = cmpl.groupby('officer_id').count()
dd = offc.merge(crid, left_on = 'officer_id', right_index = 1, how = 'outer')
Y=dd['crid']

#Y_rank = pd.get_dummies(Y['rank'])

#X2=X[X.columns[np.arange(5,13)]]
#A=X2.sum(axis=1)
#A.equals(Y)

#Y2= pd.concat([A,Y],axis=1)
#np.where(A!=Y)

clf = linear_model.LinearRegression()

predicts = cross_val_predict(clf, X, Y, cv=3)

plt.scatter(Y,predicts, s=100)
mean_squared_error(Y, predicts)
correlation = np.corrcoef(Y, predicts)[0,1]
print(correlation)



#
clf = GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,min_samples_split=5,
        min_samples_leaf=15,max_depth=5,max_features='sqrt',random_state=12,
        subsample =0.8,verbose=0,warm_start = True)

predicts = cross_val_predict(clf, X, Y, cv=3)

#clf.fit(X2,Y)
#predicts = clf.predict(X2)

mean_squared_error(Y, predicts)
plt.plot(Y,predicts,'*')
correlation = np.corrcoef(Y, predicts)[0,1]
print(correlation)


# St
from processing import *
import chris_rasmus_graph_features as gf
from count_lagged_complaints import *


scale_days = 365
last_train_year=2012
base_year = last_train_year
base_month = 12
base_day = 31
complaint_df = add_lag_to_complaints(cmpl, scale_days, base_year, base_month, base_day)

complaint_df["LAG"] = np.random.randint(0,3,13840)



# build bipartite graph
G = gf.build_bipartite_graph(complaint_df)

lag = 4
officer_ids = [int(v) for v in offc['officer_id'].unique().tolist()]
deg_thresh = 5

complaint_df.head()
complaint_df.dtypes


# get feature dictionaries
num_nbr_complaints_dict = gf.num_of_nbr_complaints(G, officer_ids, lag)
num_high_offenders = gf.num_high_offender_nbrs(G, officer_ids, deg_thresh)

type(officer_ids[0])


A = np.zeros((offc.shape[0], lag+1))
columns = ['lag_%d' % i for i in range(lag) ] + ['high_offender_nbrs']
new_df = pd.DataFrame(A,columns)
 single_merged = new_df.merge(offc['officer_ids'])
 
 
 
 
A=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in num_nbr_complaints_dict.items()])).transpose()
A.columns = ['cmpl0','cmpl1','cmpl2','cmpl3']

B=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in num_high_offenders.items()])).transpose() 
B.columns = ['num_high_offndr']

r_A = r.merge(A, left_on = 'officer_id', right_index = 1, how = 'outer')
r_AB = r_A.merge(B, left_on = 'officer_id', right_index = 1, how = 'outer')
r_AB.fillna(0)