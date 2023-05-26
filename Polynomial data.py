# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:22:05 2023

@author: user
"""

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv(r"D:\Data Science\emp_sal.csv")
X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'orange')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'black')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[5.5]])


lin_reg_2.predict(poly_reg.fit_transform([[10]]))