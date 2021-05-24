# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:40:16 2021

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

maaslar = pd.read_csv("maaslar.csv")

x=maaslar.iloc[:,1:2]
y=maaslar.iloc[:,2:]

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)
X=x.values
Y=y.values

plt.scatter(X,Y, color = 'red')
plt.plot(x,lin_reg.predict(X), color='blue')

