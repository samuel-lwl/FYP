# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:59:36 2019

@author: Samuel
"""
"""
replaced cvxopt with scipy
saved v_prior for calculating true demand. problem of x0 = 0 is still present
havent implement bayesian updating
need to check code again for mnl part
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from math import sqrt
from math import log

#x = pd.read_excel("C:/Uninotes/FYP/data/selected-sales data_children%27s book_every 99 cut 50.xlsx") # desktop
x = pd.read_excel("C:/Users/Samuel/Desktop/uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx") # laptop

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
v_mean_true = productprice
v_cov_true = np.identity(numvars)

# Generate num_data data points as prior data
num_data = 1

# Array to store data
data_demand = np.zeros((num_data, numvars))
#data_demand2 = np.zeros((num_data, numvars))

for k in range(num_data):
    # Generate a large number of V for each product since we want to approximate
    # integration by using summation
    approx = 1000
    v_prior = np.zeros((approx, numvars)) 

    for p in range(numvars): 
        for i in range(approx):
            # Perform rejection sampling, define variables
            c = sqrt(2*math.exp(1)/math.pi) 
            y = np.random.exponential() 
            u = np.random.uniform() 
            
            # If u <= the value, accept it. otherwise, reject and try again
            while u > math.exp((-(y-1)*(y-1))/2): 
                y = np.random.exponential() 
                u = np.random.uniform() 
            
            # Accept 
            z = y # ok
            
            # To decide z = z or z = -z
            u = np.random.uniform() 
            if u > 0.5:
                z = -z 
            
            # Convert to variable under v_mean_true and v_cov_true
            x_true = z * 1 + v_mean_true[p] # *1 since cov is an identity matrix 
            
            v_prior[i,p] = x_true 
    
    # here v_prior should be a full matrix
    # Calculate true demand using MNL model
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
        
        
#        temp = np.zeros(approx)
#        for app in range(approx):
#            up = math.exp(v_prior[app,j] - productprice[j,0])
#            down = 1
#            for var in range(numvars):
#                down += math.exp(v_prior[app,var] - productprice[var,0])
#            temp[app] = up/down
#        data_demand2[k,j] = np.mean(temp)

# x0
x0 = 1 - np.sum(data_demand, axis=1)








# To obtain prior distribution of V
data_v = np.zeros((num_data, numvars)) #all overestimate?
for i in range(num_data):
    # Use formula: P_j = V_j - log(X_j) + log(X_0)
    temp = productprice.flatten() + np.log(data_demand[i,:]) - math.log(x0[i])
    data_v[i] = temp

# Initialise parameters for prior distribution of V
v_mean_prior = np.mean(data_v, axis=0)
v_cov_prior = 0
for i in range(data_v.shape[0]):
    temp1 = np.reshape((data_v[i] - v_mean_prior), (numvars,1))
    temp2 = np.reshape((data_v[i] - v_mean_prior), (1,numvars))
    v_cov_prior += np.matmul(temp1, temp2)
# divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
v_cov_prior /= data_v.shape[0]

# Set v_cov_prior to be I
v_cov_prior = 0.1*np.identity(numvars)

# To keep price history, repeat num_data times for our prior data
data_prices = productprice.T
for i in range(num_data-1):
    data_prices = np.append(data_prices, productprice.T, axis=0)


iterations = 100
revenue_utility = 0
revenue_utility_basket = np.zeros(iterations)

counter = 0
collect1 = []
collect2 = []
collectx = []
collecth = []

for one in range(iterations):
    # To obtain v-hat
    v_estimate = np.random.multivariate_normal(v_mean_prior.flatten(), v_cov_prior, 1)
    
    """ use scipy.optimise """
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    
    V = np.append([0],[v_estimate])
    x_guess = np.append([x0[-1]],[data_demand[-1]])
    # Objective function, multiply by -1 since we want to maximize
    def eqn7(x):
        if (x <= 0).any(): # Return a large number if any x are <= 0
            return 1e10            
        return -1.0*(np.sum(V*x) - np.sum(x[1:]*np.log(x[1:])) + (1-x[0])*(math.log(x[0])))
    
    # Initial guess is previous demand
    bounds = Bounds(np.zeros(numvars+1), np.ones(numvars+1))
    # Constraint
    def con(t):
        return np.sum(t) - 1
    cons = {'type':'eq', 'fun': con}
    
    def grad(t):
        part1 = -((1-t[0])/t[0] - math.log(t[0]))
        part2 = -(V[1:] - 1 - np.log(t[1:]))
        ans = np.append([part1],[part2])        
        return ans
    
    def hes(t):
        ans = np.zeros((numvars+1, numvars+1))
        ans[0,0] = -(- (t[0]**-2) - (t[0]**-1))
        
        for a in range(1, numvars+1):
            ans[a,a] = -(-t[a]**-1)
        return ans
        
    opresult = minimize(eqn7, x_guess.flatten(), bounds=bounds, constraints=cons, jac=grad, hess=hes)
    newdemand_sp = opresult.x
    newdemand_sp = np.reshape(newdemand_sp, (numvars+1,1))
    
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
#        if x is None: return 0, matrix(1/(numvars+1), (numvars+1,1)) 
        if x is None: return 0, matrix(x_guess)

        # products' demand cannot be negative
        if min(x) < 0: return None
        
        # Objective function, negative sign since we want to maximise
        f = -(V*x - x[1:].T*log(x[1:]) + (1-x[0])*log(x[0]))
        
        # Vector of first partial derivatives
#        Df = -(V.T - 1 - log(x)) # (numvars+1, 1)
#        Df[0] = -((1-x[0])/x[0] - log(x[0]))

        
        Df1 = -((1-x[0])/x[0] - log(x[0]))
        Df2 = -(V[1:] - 1 - log(x[1:]))
        Df = matrix([Df1, Df2])

        global collect1
        global collect2
        global collectx
        collect1.append(np.array(Df1))
        collect2.append(np.array(Df2))
        collectx.append(np.array(x))
        
        
        if z is None: return f, Df.T
        
        # Hessian
        global H
        H = -(-x**-1)
        H[0] = -(- (x[0]**-2) - (x[0]**-1))
        
        H = spdiag(z[0] * H)
        
        collecth.append(np.array(H))
        global counter
        counter += 1

        return f[0], Df.T, H
    sol = solvers.cp(F, A=A, b=b) 
    newdemand = sol['x']
    newdemand = np.array(newdemand)

    # Find optimal price 
    newprice_cv = v_estimate.T - np.log(newdemand[1:]) + np.log(newdemand[0])

    newprice_sp = v_estimate.T  - np.log(newdemand_sp[1:]) + np.log(newdemand_sp[0])

    # Calculate true demand using MNL model and new price
    true_demand_cv = np.zeros((1,numvars))
    for j in range(numvars):
        # Numerator and denominator of MNL formula, using new price
        numerator = np.exp(v_prior[:,j] - newprice_cv[j])
        
        # For denominator, each product must be subtracted with its price, thats why 
        # we need an additional loop to loop over all products.
        # Use np.copy() since denom = v_prior would be similar to passing by reference
        denom = np.copy(v_prior)
        for i in range(numvars):
            denom[:,i] = np.exp(denom[:,i] - newprice_cv[i,0])
        
        # +1 to account for x0 where the customer buys nothing
        denom = 1 + np.sum(denom, axis=1)
        true_demand_cv[0,j] = np.mean(numerator/denom)

    # Calculate true demand using MNL model and new price
    true_demand_sp = np.zeros((1,numvars))
    for j in range(numvars):
        # Numerator and denominator of MNL formula, using new price
        numerator = np.exp(v_prior[:,j] - newprice_sp[j])
        
        # For denominator, each product must be subtracted with its price, thats why 
        # we need an additional loop to loop over all products.
        # Use np.copy() since denom = v_prior would be similar to passing by reference
        denom = np.copy(v_prior)
        for i in range(numvars):
            denom[:,i] = np.exp(denom[:,i] - newprice_sp[i,0])
        
        # +1 to account for x0 where the customer buys nothing
        denom = 1 + np.sum(denom, axis=1)
        true_demand_sp[0,j] = np.mean(numerator/denom)

    # x0 and V
    true_x0_sp = 1 - np.sum(true_demand_sp)
    true_x0_cv = 1 - np.sum(true_demand_cv)
    haha








    true_v = newprice + np.log(true_demand.T) + log(true_x0)
    
    # Calculate revenue
    revenue_utility += np.matmul(true_demand, newprice)[0,0]
    revenue_utility_basket[one] = np.matmul(true_demand, newprice)[0,0]
    
    # Append to our records
    data_prices = np.append(data_prices, newprice.T, axis=0)
    data_demand = np.append(data_demand, true_demand, axis=0)
    data_v = np.append(data_v, true_v.T, axis=0)
    
    # Recalculate parameters
    v_mean_prior = np.mean(data_v, axis=0)
    v_cov_prior = 0
    for i in range(data_v.shape[0]):
        temp1 = np.reshape((data_v[i] - v_mean_prior), (numvars,1))
        temp2 = np.reshape((data_v[i] - v_mean_prior), (1,numvars))
        v_cov_prior += np.matmul(temp1, temp2)
    # divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
    v_cov_prior /= data_v.shape[0]


    
    
    








