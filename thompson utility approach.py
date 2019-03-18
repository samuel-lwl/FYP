# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:59:36 2019

@author: Samuel
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from math import sqrt
from math import log

x = pd.read_excel("C:/Uninotes/FYP/data/selected-sales data_children%27s book_every 99 cut 50.xlsx") # desktop
#x = pd.read_excel("C:/Users/Samuel/Desktop/uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx") # laptop

# removed brand_id, agio_cut_price and free_cut_price
dataoriginalprice = x.iloc[:,[1,17]]
datafullcut = x.iloc[:,[1,19]]

# merge based on order id for cut_price
datafullcut = datafullcut.groupby(['order_id']).sum().reset_index()

# merge based on order id for price per order
dataoriginalprice = dataoriginalprice.groupby(['order_id']).sum().reset_index()

# merge both together, sort by order amount
datamerge = pd.merge(dataoriginalprice,datafullcut)
datamerge = datamerge.sort_values(by='goods_money')

# remove outlier index 1139 where cut_goods_money = 0.35
datamerge.drop(1139,inplace=True)


# changes goods_money to reflect price of just 1 quantity
    
# remove outlier index 2066 where cut_goods_money = 0.35
x.drop(2066,inplace=True)

# sort by goods_id and reset index
productprice=x.iloc[:,[5,17,13]]
productprice.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
productprice.reset_index(level=None, drop=True, inplace=True)

# note: there are 3 products in 3 orders with 0 quantity ordered cos they were returned.
# here, just obtaining price, qty ordered not important
for i, row in productprice.iterrows():
    if row["goods_amount"]==2:
        productprice.at[i, "goods_money"] = productprice.at[i, "goods_money"]/2
        
#remove goods_amount 
productprice=productprice.iloc[:,[0,1]]

numvars = x.nunique()[5]
# x.nunique() tells us there are 66 unique goods   
# obtain price for these 66 goods     
productprice.drop_duplicates(subset="goods_id", keep='first', inplace=True)
productprice.reset_index(level=None, drop=True, inplace=True)

productprice = productprice.iloc[:,1].values.reshape((numvars,1))








# True distribution of V (MVN distribution)
vmean = productprice
vcov = np.identity(numvars)

# Generate num_data data points as prior data
num_data = 20

# Array to store data
data_demand = np.zeros((num_data, numvars))

for k in range(num_data):
    # Generate a large number of V for each product since we want to approximate
    # integration by using summation
    approx = 1000
    v_prior = np.empty((approx, numvars))
    for i in range(approx):
        v_prior[i] = np.random.multivariate_normal(vmean.flatten(), vcov, 1)
    
        
    for j in range(numvars):
        numerator = np.exp(v_prior[:,j] - productprice[j])
        
        # For denominator, each product must be subtracted with its price, thats why 
        # we need an additional loop to loop over all products.
        # Use np.copy() since denom = v_prior would be similar to passing by reference
        denom = np.copy(v_prior)
        for i in range(numvars):
            denom[:,i] = np.exp(denom[:,i] - productprice[i])
        
        # +1 to account for x0 where the customer buys nothing
        denom = 1 + np.sum(denom, axis=1)
        data_demand[k,j] = np.mean(numerator/denom)

# x0
x0 = 1 - np.sum(data_demand, axis=1)








# To calculate estimates of V
data_v = np.zeros((num_data, numvars)) #all overestimate?
for i in range(num_data):
    temp = productprice.flatten() + np.log(data_demand[i,:]) - math.log(x0[i])
    data_v[i] = temp

# Initialise parameters for prior distribution of V
v_mean = np.mean(data_v, axis=0)
v_cov = 0
for i in range(data_v.shape[0]):
    temp1 = np.reshape((data_v[i] - v_mean), (numvars,1))
    temp2 = np.reshape((data_v[i] - v_mean), (1,numvars))
    v_cov += np.matmul(temp1, temp2)
# divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
v_cov /= data_v.shape[0]


v_estimate = np.random.multivariate_normal(v_mean.flatten(), v_cov, 1)


from cvxopt import solvers, matrix, spdiag, log, exp, div, mul



A, b = matrix(1.0, (1, numvars+1)), matrix(1.0)
V = matrix(v_estimate, (numvars,1))
# Adding in v0 = 0
V = matrix([0, V])
V = V.T

# maximize   V*x + x[1:].T*matrix(np.log(x[1:])) + (1 - x[0])*math.log(x[0])
# subject to A*x = b

def F(x=None, z=None):
    # Set sample point x0 as 1/(number of variables)
    if x is None: return 0, matrix(1/(numvars+1), (numvars+1,1))
    
    # products' demand cannot be negative
    if min(x) < 0: return None
    
    # Objective function, negative sign since we want to maximise
    f = -(V*x - x[1:].T*log(x[1:]) + (1-x[0])*log(x[0]))
    
    # Vector of first partial derivatives
    grad = V.T + mul(x, log(x))  
    grad[0] = (1-x[0])/x[0] - log(x[0])
    if z is None: return f, grad.T
    
    # Hessian, why only z[0]?
    H = - 1 - log(x)
    H[0] = -(x[0]**-2)
    H = spdiag(z[0] * H)
    return f, grad.T, H
sol = solvers.cp(F, A=A, b=b) 
p = sol['x']















    
    
    








