# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:09:58 2021

@author: User
"""

import numpy as np
import pandas as pd

raw_tenis = pd.read_csv("odev_tenis.csv")
tenis = raw_tenis
# Veri Temizliği
from sklearn import preprocessing

# tenis2 = tenis.apply(preprocessing.LabelEncoder().fit_transform)
# Bütün tabloya LabelEncoder uygulamanın kolay yolu

outlook = tenis.iloc[:,0:1].values
         
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(tenis.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
outlook = pd.DataFrame(outlook, columns=["overcast","rainy","sunny"])

tenis["windy"] = (pd.get_dummies(tenis["windy"])).iloc[:,1]
tenis["play"] = (pd.get_dummies(tenis["play"])).iloc[:,1]

tenis = pd.concat([outlook,tenis.iloc[:,1:]],axis=1)

# Model Kurma
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

left = tenis.iloc[:,0:3]
right = tenis.iloc[:,4:]
temperature = tenis.iloc[:,3]

tenis_test = pd.concat([left,right], axis=1)

x_train, x_test, y_train, y_test = train_test_split(tenis_test,temperature, test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

temp_pred = regressor.predict(x_test)

# Backward Elimination

ttest = np.append(arr=np.ones((14,1)).astype(int), values = tenis_test, axis=1)

ttest_l = tenis_test.iloc[:,[0,1,2,3,4,5]].values
ttest_l = np.array(ttest_l, dtype=float)
model = sm.OLS(temperature, ttest_l).fit()
print(model.summary())

