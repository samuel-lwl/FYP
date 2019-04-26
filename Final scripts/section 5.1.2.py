import pandas as pd
# =============================================================================
# use dataset
# =============================================================================
x = pd.read_excel("selected-sales data_children%27s book_every 99 cut 50.xlsx") 

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

beta = 0.5 
price0 = prices[1]

# Initial demand
f1 = np.random.uniform(0.5,5,numvars)

c0 = 0.05 

noisemean = np.zeros(numvars)
noisecov = np.identity(numvars)

data_ts = np.array(day01.iloc[:,1])
f1 = data_ts*beta + c0 + np.random.multivariate_normal(noisemean, noisecov, 1)
# f1 cannot be negative
while (f1<0).any():
        f1 = data_ts*beta + c0 + np.random.multivariate_normal(noisemean, noisecov, 1)
data_ts = np.reshape(data_ts, (1,numvars))
 
prevprice = price0

# To store histories of f, prevprice, observedx, elastmean
tripledata_ts = []
revenue_basket_ts = []

datapts = 16

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

# 1500 stock for each product
stock_ts = np.ones((numvars, 1))
stock_ts *= 1500

i = 0
f0 = data_ts
f0 = np.reshape(f0, (numvars,1))

# Data generating
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
    
    # Random sample for elasticities
    elast_ts = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
    # Ensures that all components are negative 
    while (elast_ts<0).all() == False: 
        print("elast is positive")
        print(i)
        elast_ts = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
    elast_ts = np.reshape(elast_ts, (numvars,1))
        
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
    observedx = f * (newprice / prevprice)**gammastar + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k,0] < 0:
            observedx[k,0] = 0
    
    # Update stocks
    stock_ts -= observedx
    
    # Append new data    
    data_ts = np.append(data_ts, np.transpose(observedx), axis=0)

    # Add data as triplet into tripledata
    tripledata_ts.append([f, newprice, observedx, elastmean])
    revenue_basket_ts.append(np.sum(np.multiply(observedx, newprice)))
    
    # For M inverse matrix
    thet = np.multiply(np.reshape(newprice**2,(numvars,1)), f)
    thet = np.divide(thet, prevprice)
    thet = thet - np.multiply(np.reshape(newprice,(numvars,1)), f)
    minv = (thet*thet.T)/sighat**2 + 1e-5*np.identity(numvars) 
    
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

priceshistory_ts = []
for i in range(len(tripledata_ts)):
    priceshistory_ts.append(tripledata_ts[i][1])

fhistory_ts = []
for i in range(len(tripledata_ts)):
    fhistory_ts.append(tripledata_ts[i][0])
    
obshistory_ts = []
for i in range(len(tripledata_ts)):
    obshistory_ts.append(tripledata_ts[i][2])

plt.plot(np.cumsum(revenue_basket_ts),'ko-')

# =============================================================================
# constant price
# =============================================================================

# To store histories of f, prevprice, observedx, elastmean
tripledata_constant = []
revenue_basket_constant = []

prevprice = price0

data_constant = data_ts[0,:]
data_constant = np.reshape(data_constant, (1, numvars))

# 1500 stock for each product
stock_constant = np.ones((numvars,1))
stock_constant *= 1500

i = 0
f = data_constant

# Data generating
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
            f += (beta**j)*np.reshape(data_constant[i-j], (numvars,1))
                
    f = np.reshape(f, (numvars,1))
     
    # Generate price
    newprice = prevprice
    
    # Observed demand
    observedx = f + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k,0] < 0:
            observedx[k,0] = 0
    
    # Update stocks
    stock_constant -= observedx

    # Append new data    
    data_constant = np.append(data_constant, np.transpose(observedx), axis=0)

    # Add data as triplet into tripledata
    tripledata_constant.append([f, newprice, observedx])
    revenue_basket_constant.append(np.sum(np.multiply(observedx, newprice)))
        
    # Update prevprice to new price
    prevprice=price0

#plt.plot(revenue_basket_constant)
plt.plot(np.cumsum(revenue_basket_constant),'k')

priceshistory_constant = []
for i in range(len(tripledata_constant)):
    priceshistory_constant.append(tripledata_constant[i][1])

fhistory_constant = []
for i in range(len(tripledata_constant)):
    fhistory_constant.append(tripledata_constant[i][0])
    
obshistory_constant = []
for i in range(len(tripledata_constant)):
    obshistory_constant.append(tripledata_constant[i][2])


plt.ylabel('Cumulated revenue',fontsize=20)
plt.xlabel('Time period',fontsize=20)
plt.legend(['Thompson Sampling','Constant pricing'],fontsize=20)
plt.title('TS vs Constant pricing with non-stationary demand function without inventory constraint',fontsize=20)

# assigning a cost to each product where cost = 0.5 * original price
cost = prices[1]
cost *= 0.5

timeperiod = 15
totalts = np.zeros((numvars,1))
totalcon = np.zeros((numvars,1))

# calculate how much stock has been sold
for i in range(timeperiod):
    totalts += obshistory_ts[i]
for i in range(timeperiod):
    totalcon += obshistory_constant[i]

# calculate leftover stock
left_ts = 1500 - totalts
left_con = 1500 - totalcon
revenue_basket_constant = revenue_basket_constant[:timeperiod]
revenue_basket_ts = revenue_basket_ts[:timeperiod]

# overall revenue
rev_con = np.sum(revenue_basket_constant)
rev_ts = np.sum(revenue_basket_ts)

# overall costs
cost_con = np.sum(cost*left_con)
cost_ts = np.sum(cost*left_ts)

# overall loss where loss = costs - revenue
loss_ts = cost_ts - rev_ts
loss_con = cost_con - rev_con

# overall profit where profit = -costs
profit_ts = -1 * loss_ts
profit_con = -1 * loss_con
