# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 04:39:06 2019

@author: Samuel

In this script, I try to use the data generating mechanism as a start point,
instead of using it to generate prior data.

Applied to children's dataset.

Each iteration, remove products that are already sold. dimension reduction
"""
import pandas as pd
# =============================================================================
# use dataset
# =============================================================================
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

# create 2 other price vectors
productpricelower = productprice*0.9
productpricehigher = productprice*1.1
#productpricelower = productprice-5
#productpricehigher = productprice+5

# Compile prices for each arm into one array
# 0: lower, 1: middle, 2: higher
prices = [productpricelower.iloc[:,1].values.reshape((numvars,1)), productprice.iloc[:,1].values.reshape((numvars,1)), productpricehigher.iloc[:,1].values.reshape((numvars,1))]


# All goods_id in the dataset
allgoods = x.loc[:,"goods_id"]
allgoods = allgoods.to_frame() # convert to dataframe
allgoods.drop_duplicates(keep='first', inplace=True)
allgoods.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
allgoods.reset_index(level=None, drop=True, inplace=True)

# get quantity sold for each good_id for each day
# as_index=FALSE will put goods_id as a new column instead of as index
# Choose day 1 and 4 since these two days have no missing goods
day01 = x.loc[x['order_date']==20170804,:]
day01 = day01.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
day04 = x.loc[x['order_date']==20170807,:]
day04 = day04.groupby('goods_id',as_index=False)[["goods_amount"]].sum()

# day1 573 sold, day4 768 sold. proceed with day 1.


# =============================================================================
# End
# =============================================================================



# =============================================================================
# MAX-REV-TS
# =============================================================================
numvars = 66

# Initialisation
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt


# True parameter
gammastar = np.random.uniform(-3,-1,numvars)
gammastar = np.reshape(gammastar, (numvars,1))

# Initial parameters
beta = 0.5 # original 0.5
price0 = prices[1]
c0 = 0.05 # original 0.005

# Initial demand
f1 = np.random.uniform(0.5,5,numvars)

# Noise
noisemean_true = np.zeros(numvars)
noisemean_true = np.reshape(noisemean_true, (numvars, 1))
noisecov_true = np.identity(numvars)

data_ts = np.array(day01.iloc[:,1])
f1 = data_ts*beta + c0 + np.random.multivariate_normal(noisemean_true.flatten(), noisecov_true, 1)
# f1 cannot be negative
while (f1<0).any():
        f1 = data_ts*beta + c0 + np.random.multivariate_normal(noisemean_true.flatten(), noisecov_true, 1)
data_ts = np.reshape(data_ts, (1,numvars))
# For constant approach
data_constant = np.copy(data_ts)

# Set base price as previous price 
prevprice = price0

# To store histories of f, prevprice, observedx, elastmean
tripledata_ts = []
revenue_basket_ts = []

datapts = 101

# Prior estimate of gamma, use as mean of distribution
gammaprior = np.random.uniform(-5,-1,numvars)
# constant c
c = 0.1 * np.mean(gammaprior)
elastmean = np.reshape(np.array(gammaprior),(numvars,1))
elastcov = c*np.identity(numvars)

historical_rev = np.zeros(2)
historical_rev[0] = np.matmul(data_ts, price0)
historical_rev[1] = np.matmul(np.reshape(np.array(day04.iloc[:,1]), (1,numvars)) , price0)
sighat = np.std(historical_rev)

# 5000 stock for each product
stock_ts = np.ones((numvars, 1))
stock_ts *= 50

truth = stock_ts <= 0

# make another copy for TS
gammastar_ts = gammastar
noisemean_ts = noisemean_true
noisecov_ts = noisecov_true

i = 0
# Data generating
while (stock_ts>0).any():
    i += 1
    
#for i in range(1, datapts):
    
    # Demand forecast
    if i == 1:
        f = f1
    else:
        noise = np.random.multivariate_normal(noisemean_ts.flatten(), noisecov_ts, 1)
        noise = np.reshape(noise, (numvars,1))
        f = c0 + noise

        # To calculate f
        for j in range(1, i+1):
            f += (beta**j)*np.reshape(data_ts[i-j], (numvars,1))
                
    f = np.reshape(f, (numvars,1))

    # Random sample for elasticities
    elast_ts = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
    # Ensures that all components are negative 
    while (elast_ts<0).all() == False: 
        print("elast is positive")
        print(i)
        elast_ts = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
    elast_ts = np.reshape(elast_ts, (numvars,1))
    
    # Save for max-rev-passive
#    if i == 1:
#        elast1 = elast_ts
#    if i == 2:
#        elast2 = elast_ts
        
    # Generate price
    """using scipy.optimize"""
    # Objective function, multiply by -1 since we want to maximize
    def eqn7(p):
        return -1.0*np.sum(p*p*f.flatten()*elast_ts.flatten()/prevprice.flatten() - p*f.flatten()*elast_ts.flatten() + p*f.flatten())
    
    # Initial guess is 1.05 * previous price
    bounds = Bounds(prevprice.flatten()*0.9, prevprice.flatten()*1.1)
    opresult = minimize(eqn7, prevprice.flatten()*1.05, bounds=bounds)
    newprice = opresult.x
    newprice = np.reshape(newprice, (numvars,1))   
    
    # Observed demand
    observedx = f * (newprice / prevprice)**gammastar_ts + np.reshape(np.random.multivariate_normal(noisemean_ts.flatten(), noisecov_ts, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k,0] < 0:
            observedx[k,0] = 0
        if truth[k,0] == True:
            observedx[k,0] = 0
    
    # Update stocks
    stock_ts -= observedx
    
    # Append new data    
    data_ts = np.append(data_ts, np.transpose(observedx), axis=0)

    # Add data as triplet into tripledata
    tripledata_ts.append([f, prevprice, observedx, elastmean])
    revenue_basket_ts.append(np.sum(np.multiply(observedx, newprice)))

#        if i == 1:
#            historical_rev[1] = np.sum(np.multiply(observedx, newprice))
#            sighat = np.std(historical_rev)
    
    # For M inverse matrix
    thet = np.multiply(np.reshape(newprice**2,(numvars,1)), f)
    thet = np.divide(thet, prevprice)
    thet = thet - np.multiply(np.reshape(newprice,(numvars,1)), f)
    minv = (thet*thet.T)/sighat**2 + 1e-5*np.identity(numvars) # original lambda = 1e-5
    
    # For M inverse beta matrix
    rbar = np.sum(np.multiply(np.reshape(newprice,(numvars,1)), f))
    rt = float(np.sum(np.multiply(observedx, np.reshape(newprice,(numvars,1)))))
    minvb = (rt - rbar)/(sighat**2)*thet
    
    # Update mean of elasticity's distribution
    pt1 = np.linalg.inv(np.linalg.inv(elastcov) + minv)
    pt2 = np.matmul(np.linalg.inv(elastcov), elastmean) + minvb
    elastmean = np.matmul(pt1, pt2)
            
    # Update covariance of elasticity's distribution
    elastcov = pt1
    
    # Update prevprice to new price
    prevprice = np.reshape(newprice, (numvars,1))
    
    
    
    
    # Get T/F values 
    truth = stock_ts <= 0
    # To get all indices and reshape appropriately
    rang = np.array(range(len(truth)))
    rang = np.reshape(rang, (len(truth), 1))
    # To get indices that are <= 0 since we want to remove them
    want = rang[truth]
    # Sort in descending order for cov matrices
    want_desc = want[::-1]
    # Update numvars
    numvars -= len(want)

    # Reset means. must negate truth since these are the ones we want to keep
    noisemean_ts = noisemean_ts[np.logical_not(truth)]
    noisemean_ts = np.reshape(noisemean_ts, (numvars, 1))
    elastmean = elastmean[np.logical_not(truth)]
    elastmean = np.reshape(elastmean, (numvars, 1))
    
    # Reset cov matrices
    for index in want_desc:
        # Delete row
        noisecov_ts = np.delete(noisecov_ts, index, 0)
        elastcov = np.delete(elastcov, index, 0)
        # Delete col
        noisecov_ts = np.delete(noisecov_ts, index, 1)
        elastcov = np.delete(elastcov, index, 1)
        data_ts = np.delete(data_ts, index, 1)
                
    # Reset prevprice
    prevprice = prevprice[np.logical_not(truth)]
    prevprice = np.reshape(prevprice, (numvars, 1))
    
    # Reset gammastar_ts
    gammastar_ts = gammastar_ts[np.logical_not(truth)]
    gammastar_ts = np.reshape(gammastar_ts, (numvars, 1))
    
    # Reset stock_ts
    stock_ts = stock_ts[np.logical_not(truth)]
    stock_ts = np.reshape(stock_ts, (numvars, 1))


priceshistory = []
for i in range(len(tripledata_ts)):
    priceshistory.append(tripledata_ts[i][1])

fhistory = []
for i in range(len(tripledata_ts)):
    fhistory.append(tripledata_ts[i][0])
    
obshistory = []
for i in range(len(tripledata_ts)):
    obshistory.append(tripledata_ts[i][2])
plt.plot(revenue_basket_ts)

# =============================================================================
# constant price
# =============================================================================

# To store histories of f, prevprice, observedx, elastmean
tripledata_constant = []
revenue_basket_constant = []

prevprice = price0

numvars = 66

# 5000 stock for each product
stock_constant = np.ones((numvars, 1))
stock_constant *= 50

truth = stock_constant <= 0

# Copy for constant approach
noisemean_constant = noisemean_true
noisecov_constant = noisecov_true
gammastar_constant = gammastar

i = 0
# Data generating
while (stock_constant>0).any():
    i += 1
#for i in range(1, datapts):
    # Demand forecast
    if i == 1:
        f = f1
    else:
        noise = np.random.multivariate_normal(noisemean_constant.flatten(), noisecov_constant, 1)
        noise = np.reshape(noise, (numvars,1))
        f = c0 + noise

        # To calculate f
        for j in range(1, i+1):
            f += (beta**j)*np.reshape(data_constant[i-j], (numvars,1))
                
    f = np.reshape(f, (numvars,1))
        
    # Generate price
    newprice = prevprice
    
    # Observed demand
    observedx = f + np.reshape(np.random.multivariate_normal(noisemean_constant.flatten(), noisecov_constant, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k,0] < 0:
            observedx[k,0] = 0
        if truth[k,0] == True:
            observedx[k,0] = 0
    
    # Update stocks
    stock_constant -= observedx

    # Append new data    
    data_constant = np.append(data_constant, np.transpose(observedx), axis=0)

    # Add data as triplet into tripledata
    tripledata_constant.append([f, prevprice, observedx])
    revenue_basket_constant.append(np.sum(np.multiply(observedx, newprice)))

    # Update prevprice to new price
    prevprice = np.reshape(newprice, (numvars,1))



    truth = stock_constant <= 0
    # To get all indices and reshape appropriately
    rang = np.array(range(len(truth)))
    rang = np.reshape(rang, (len(truth), 1))
    # To get indices that are <= 0 since we want to remove them
    want = rang[truth]
    # Sort in descending order for cov matrices
    want_desc = want[::-1]
    # Update numvars
    numvars -= len(want)

    # Reset means. must negate truth since these are the ones we want to keep
    noisemean_constant = noisemean_constant[np.logical_not(truth)]
    noisemean_constant = np.reshape(noisemean_constant, (numvars, 1))
    
    # Reset cov matrices
    for index in want_desc:
        # Delete row
        noisecov_constant = np.delete(noisecov_constant, index, 0)
        # Delete col
        noisecov_constant = np.delete(noisecov_constant, index, 1)
        data_constant = np.delete(data_constant, index, 1)
                
    # Reset prevprice
    prevprice = prevprice[np.logical_not(truth)]
    prevprice = np.reshape(prevprice, (numvars, 1))
    
    # Reset gammastar_constant
    gammastar_constant = gammastar_constant[np.logical_not(truth)]
    gammastar_constant = np.reshape(gammastar_constant, (numvars, 1))
    
    # Reset stock_constant
    stock_constant = stock_constant[np.logical_not(truth)]
    stock_constant = np.reshape(stock_constant, (numvars, 1))

plt.plot(revenue_basket_constant)

priceshistory_constant = []
for i in range(len(tripledata_constant)):
    priceshistory_constant.append(tripledata_constant[i][1])

fhistory_constant = []
for i in range(len(tripledata_constant)):
    fhistory_constant.append(tripledata_constant[i][0])
    
obshistory_constant = []
for i in range(len(tripledata_constant)):
    obshistory_constant.append(tripledata_constant[i][2])


plt.plot(np.cumsum(revenue_basket_ts))
plt.plot(np.cumsum(revenue_basket_constant))






# =============================================================================
# MAX-REV-PASSIVE
# =============================================================================
haha
    
prevprice = price0

# Same starting point as TS
data_passive = data_ts[0]
data_passive = np.reshape(data_passive, (1, numvars))

# To store histories of f, prevprice, observedx, elastmean
tripledata_passive = []
revenue_basket_passive = []
datapts = 101

passive_d = np.zeros((1,numvars))
passive_p = np.zeros((1,numvars))

for i in range(1, datapts):
    # Demand forecast
    if i == 1:
        f = f1
    else:
        noise = np.random.multivariate_normal(noisemean, noisecov, 1)
        noise = np.reshape(noise, (numvars,1))
        f = c0 + noise

        # To calculate f
        for j in range(1, i+1):
            f += (beta**j)*np.reshape(data_ts[i-j], (numvars,1))
                
    f = np.reshape(f, (numvars,1))
    
    # elasticity estimate 
    if i == 1:
        elast_passive = elast1
    if i == 2:
        elast_passive = elast2
    else:
        dbar = np.mean(passive_d, axis=0)
        pbar = np.mean(passive_p, axis=0)
        elast_passive = np.zeros((numvars,1))
        for var in range(numvars):
            top = np.sum( (passive_p[:,var] - pbar[var]) * (passive_d[:,var] - dbar[var]) )
            bottom = np.sum( (passive_p[:,var] - pbar[var]) * (passive_p[:,var] - pbar[var]) )
            elast_passive[var] = top/bottom
        
    """using scipy.optimize"""
    # Objective function, multiply by -1 since we want to maximize
    def eqn7(p):
        return -1.0*np.sum(p*p*f.flatten()*elast_passive.flatten()/prevprice.flatten() - p*f.flatten()*elast_passive.flatten() + p*f.flatten())
    
    # Initial guess is 1.05 * previous price
    bounds = Bounds(prevprice.flatten()*0.9, prevprice.flatten()*1.1)
    opresult = minimize(eqn7, prevprice.flatten()*1.05, bounds=bounds)
    newprice = opresult.x
    newprice = np.reshape(newprice, (numvars,1))   
    
    # Observed demand
    observedx = f * (newprice / prevprice)**gammastar + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k,0] < 0:
            # cannot set to 0 otherwise log(0) will give error
            observedx[k,0] = 0.001
    
    # Append new data    
    data_passive = np.append(data_passive, np.transpose(observedx), axis=0)

    # Add data as triplet into tripledata
    tripledata_passive.append([f, prevprice, observedx, elast_passive])
    revenue_basket_passive.append(np.sum(np.multiply(observedx, newprice)))
    
    # Append to our history of demand and price ratios for OLS
    if i == 1:
        passive_d = np.log(observedx.T)
        passive_p = np.log((newprice/prevprice).T)
    else:
        passive_d = np.append(passive_d, np.log(observedx.T), axis=0)
        passive_p = np.append(passive_p, (newprice/prevprice).T, axis=0)
    
    # Update prevprice to new price
    prevprice = np.reshape(newprice, (numvars,1))

elast_passivehistory = []
for i in range(len(tripledata_passive)):
    elast_passivehistory.append(tripledata_passive[i][3])


plt.plot(revenue_basket_passive)

