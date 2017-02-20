# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:27:33 2017

@author: Mehraveh
"""

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

%matplotlib inline
python_dir = '/Users/Mehraveh/Desktop/yins-datahack-2017'
os.chdir(python_dir)




from count_lagged_complaints import *
from chris_rasmus_graph_features import *
from processing import *
%matplotlib inline

offc = pd.read_csv('/Users/Mehraveh/Desktop/policy_lab/toy.officer_data.csv',sep=',')
cmpl = pd.read_csv('/Users/Mehraveh/Desktop/policy_lab/toy.complaint_data.csv', sep=',')


scale_days = 365
last_train_year = 2012
base_year = last_train_year
base_month = 12
base_day = 31
lag_max = 4
deg_thresh = 5

cmpl_severity = add_complaint_severity(cmpl)
r = add_complaints_by_year_and_severity(last_train_year, lag_max, cmpl_severity, offc) 


complaint_df = add_lag_to_complaints(cmpl_severity, scale_days, base_year, base_month, base_day)


# build bipartite graph
G = build_bipartite_graph(complaint_df)
officer_ids = offc['officer_id'].unique()


# get feature dictionaries
num_nbr_complaints_dict = num_of_nbr_complaints(G, officer_ids, lag_max)
num_high_offenders = num_high_offender_nbrs(G, officer_ids, deg_thresh)
num_nbr_complaints_dict_pf = num_of_nbr_complaints_past_future(G, officer_ids, lag_max)


A=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in num_nbr_complaints_dict.items()])).transpose()
A.columns = ['cmpl0','cmpl1','cmpl2','cmpl3']

B=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in num_high_offenders.items()])).transpose() 
B.columns = ['num_high_offndr']
C=pd.DataFrame(dict([(k,pd.Series(v)) for k,v in num_nbr_complaints_dict_pf.items()])).transpose()
C.columns = ['cmpl0_p','cmpl1_p','cmpl2_p','cmpl3_p','cmpl0_f','cmpl1_f','cmpl2_f','cmpl3_f']


r_A = r.merge(A, left_on = 'officer_id', right_index = 1, how = 'outer')
r_AB = r_A.merge(B, left_on = 'officer_id', right_index = 1, how = 'outer')
r_AB = r_AB.fillna(0);
r_C = r_AB.merge(C, left_on = 'officer_id', right_index = 1, how = 'outer')
r_C.head()
r_C.columns

# Making X vector (features)
# X = r_AB
X=r

#Dropping "FirstName", "LastName", "BirthYear","Rank"
X = X.drop(X.columns[[0,1,5,7,11]], axis=1)
# Getting years from apponted dates and normalizing
# temp = X[X.columns[0]].astype(str).str[0:4].astype(int)
# X[X.columns[0]] = (temp-np.mean(temp))/np.std(temp)
X[X.columns[0]] = X[X.columns[0]].astype(str).str[0:4].astype(int)


#Encoding categorical variables
X_cat = X[X.columns[[1,2]]]
XX=X_cat.apply(LabelEncoder().fit_transform)
X[X.columns[[1,2]]] = XX
np.where(np.isnan(X))
X_race = pd.get_dummies(X['race'])
X_race.columns = ['race0','race1','race2','race3']
X = X.drop(X.columns[[1]], axis=1)
X = X.join(X_race, how='outer')

X.head()
X.columns


#X = X.drop(X.columns[[0,1,2,3,4,5,12,13,14,15,16]], axis=1)

# Making Y vector

Y = add_complaints_by_year(2015, 2, cmpl, offc) 
Y=Y[[Y.columns[Y.shape[1]-1],Y.columns[Y.shape[1]-2]]]
Y=Y.sum(axis=1)

# LINEAR model


clf = linear_model.LinearRegression()
predicts = cross_val_predict(clf, X, Y, cv=3)

clf.fit(X,Y)
predicts = clf.predict(X)

plt.scatter(Y,predicts, s=100)
mean_squared_error(Y, predicts)
correlation = np.corrcoef(Y, predicts)[0,1]
print(correlation)



# LINEAR model


A=clf.coef_

