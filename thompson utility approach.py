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







""" code starts from here """
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
        # Numerator and denominator of MNL formula
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
    # Use formula: P_j = V_j - log(X_j) + log(X_0)
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

# To obtain v-hat
v_estimate = np.random.multivariate_normal(v_mean.flatten(), v_cov, 1)

""" use scipy.optimise """
#from scipy.optimize import minimize
#from scipy.optimize import Bounds
#
#V = np.append([0],[v_estimate])
#x_guess = np.append([x0[-1]],[data_demand[-1]])
## Objective function, multiply by -1 since we want to maximize
#def eqn7(x):
#    return -1.0*(np.sum(V*x) - np.sum(x[1:]*np.log(x[1:])) + (1-x[0])*(math.log(x[0])))
#
## Initial guess is previous demand
#bounds = Bounds(x_guess.flatten()*0, np.ones(numvars+1))
## Constraint
#def con(t):
#    return np.sum(t) - 1
#cons = {'type':'eq', 'fun': con}
#opresult = minimize(eqn7, x_guess.flatten(), bounds=bounds, constraints=cons)
#newdemand = opresult.x

""" use cvxopt """
from cvxopt import solvers, matrix, spdiag, log, exp, div, mul
solvers.options['show_progress'] = False

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
    global Df
    Df = -(V.T + mul(x, log(x)))  
    Df[0] = -((1-x[0])/x[0] - log(x[0]))
    if z is None: return f, Df.T
    
    # Hessian
    global H
    H = -(- 1 - log(x))
    H[0] = -(- (x[0]**-2) - (x[0]**-1))
    H = spdiag(z[0] * (-H))
    
    return f, Df.T, H
sol = solvers.cp(F, A=A, b=b) 
p = sol['x']
""" ValueError: Rank(A) < p or Rank([H(x); A; Df(x); G]) < n """

# Find optimal price 
price = v_estimate.T + np.log(p[1:]) + log(p[0])



# example from cvxopt website
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div
solvers.options['show_progress'] = False

# minimize     p'*log p
# subject to  -0.1 <= a'*p <= 0.1
#              0.5 <= (a**2)'*p <= 0.6
#             -0.3 <= (3*a**3 - 2*a)'*p <= -0.2
#              0.3 <= sum_{k:ak < 0} pk <= 0.4
#              sum(p) = 1
#
# a in R^100 is made of 100 equidistant points in [-1,1].
# The variable is p (100).

n = 100
a = -1.0 + (2.0/(n-1)) * matrix(list(range(n)), (1,n))
I = [k for k in range(n) if a[k] < 0]
G = matrix([-a, a, -a**2, a**2, -(3 * a**3 - 2*a), (3 * a**3 - 2*a),
    matrix(0.0, (2,n))])
G[6,I] = -1.0
G[7,I] =  1.0
h = matrix([0.1, 0.1, -0.5, 0.6, 0.3, -0.2, -0.3, 0.4 ])

A, b = matrix(1.0, (1,n)), matrix(1.0)

# minimize    x'*log x
# subject to  G*x <= h
#             A*x = b
#
# variable x (n).

def F(x=None, z=None):
   if x is None: return 0, matrix(1.0, (n,1))
   if min(x) <= 0: return None
   f = x.T*log(x)
   global grad
   grad = 1.0 + log(x)
   if z is None: return f, grad.T
   global H
   H = spdiag(z[0] * x**-1)
   return f, grad.T, H
sol = solvers.cp(F, G, h, A=A, b=b)
p = sol['x']



    
    
    








