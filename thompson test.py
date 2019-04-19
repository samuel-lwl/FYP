# -*- coding: utf-8 -*-

"""
Shouldnt apply dynamic pricing to MNL? cos elasticity will
always have some positive. because we need all of X_i to sum up
to 1 so for products that have increase in demand, other products
must decrease in demand to satisfy the constraint. this means
we will always have some products that have positive elasticity.
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

# goods_money is the amount a customer paid, quantity * price


# =============================================================================
# checking for full-cut pricing
# =============================================================================
#%matplotlib qt

mylist=[]
# histogram using 99 as threshold 
##datamerge.hist(column="goods_money",bins=[0,99,99*2,99*3,99*4,99*5,99*6,99*7])
for i in range(1,8): 
    # calculate mean of full_cut for each range 
    mylist.append(datamerge.loc[(datamerge["goods_money"]>=99*(i-1)) & (datamerge["goods_money"]<99*i),:].mean()[2])

mylist2=[]    
# histogram using 80 as threshold 
##datamerge.hist(column="goods_money",bins=[0,80,80*2,80*3,80*4,80*5,80*6,80*7,80*8,80*9])
for i in range(1,8): 
    # calculate mean of full_cut for each range 
    mylist2.append(datamerge.loc[(datamerge["goods_money"]>=80*(i-1)) & (datamerge["goods_money"]<80*i),:].mean()[2])

mylist3=[]
# histogram using 120 as threshold 
##datamerge.hist(column="goods_money",bins=[0,120,120*2,120*3,120*4,120*5,120*6,120*7])
for i in range(1,8): 
    # calculate mean of full_cut for each range 
    mylist3.append(datamerge.loc[(datamerge["goods_money"]>=120*(i-1)) & (datamerge["goods_money"]<120*i),:].mean()[2])

mylist4=[]
# histogram using 110 as threshold 
##datamerge.hist(column="goods_money",bins=[0,110,110*2,110*3,110*4,110*5,110*6,110*7])
for i in range(1,8): 
    # calculate mean of full_cut for each range 
    mylist4.append(datamerge.loc[(datamerge["goods_money"]>=110*(i-1)) & (datamerge["goods_money"]<110*i),:].mean()[2])

mylist5=[]
# histogram using 90 as threshold 
##datamerge.hist(column="goods_money",bins=[0,90,90*2,90*3,90*4,90*5,90*6,90*7,90*8])
for i in range(1,8): 
    # calculate mean of full_cut for each range 
    mylist5.append(datamerge.loc[(datamerge["goods_money"]>=90*(i-1)) & (datamerge["goods_money"]<90*i),:].mean()[2])

plt.plot(mylist2,'k--', mylist5,'k-.', mylist,'ko-', mylist4,'k:', mylist3, 'k+-.')
plt.ylabel('Mean discount',fontsize=20)
plt.xlabel('Groups',fontsize=20)
plt.legend(['Threshold = 80','Threshold = 90','Threshold = 99','Threshold = 110', 'Threshold = 120'],fontsize=20)
plt.title("Verifying full-cut promotion",fontsize=20)
plt.show()
ha
#plt.plot(np.cumsum(basket_cTS),'ko-')
#plt.plot(np.cumsum(basket_eg),'k--')
#plt.plot(np.cumsum(basket_egd),'k-.')
#plt.plot(np.cumsum(basket_egoi),'k:')
#plt.plot(np.cumsum(basket_ucb),'k+-.')
#plt.plot(np.cumsum(basket_ucbt),'kx:')
#plt.ylabel('Cumulated revenue',fontsize=20)
#plt.xlabel('Time period',fontsize=20)
#plt.legend(['classical Thompson sampling','epsilon-greedy','epsilon-greedy w/ decay','epsilon-greedy w/ optimistic initialisation','ucb1','ucb1-tuned'],fontsize=20)
#plt.title("Comparing TS with other algorithms",fontsize=20)
#plt.show()
###############################################################################################################################################
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

##########################################################################################
######################################### now trying to use mle for first price vector
## 11 days, take as 11 observations for each good
#
## All goods_id in the dataset
#allgoods = x.loc[:,"goods_id"]
#allgoods = allgoods.to_frame() # convert to dataframe
#allgoods.drop_duplicates(keep='first', inplace=True)
#allgoods.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#allgoods.reset_index(level=None, drop=True, inplace=True)
#
## get quantity sold for each good_id for each day
## as_index=FALSE will put goods_id as a new column instead of as index
#day1 = x.loc[x['order_date']==20170804,:]
#day1 = day1.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day2 = x.loc[x['order_date']==20170805,:]
#day2 = day2.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day3 = x.loc[x['order_date']==20170806,:]
#day3 = day3.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day4 = x.loc[x['order_date']==20170807,:]
#day4 = day4.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day5 = x.loc[x['order_date']==20170808,:]
#day5 = day5.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day6 = x.loc[x['order_date']==20170809,:]
#day6 = day6.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day7 = x.loc[x['order_date']==20170810,:]
#day7 = day7.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day8 = x.loc[x['order_date']==20170811,:]
#day8 = day8.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day9 = x.loc[x['order_date']==20170812,:]
#day9 = day9.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day10 = x.loc[x['order_date']==20170813,:]
#day10 = day10.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#day11 = x.loc[x['order_date']==20170814,:]
#day11 = day11.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
#
## checking for days where not all goods were sold, because in previous step if a good wasnt sold on
## a particular day, it will not show up in the dataframe.
## concat both master list and dayX, choose only goods_id column, convert to dataframe and drop duplicates
## because duplicates mean they appear in both the master list and the list of goods sold that day
#day1missing=(pd.concat([allgoods,day1],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day2missing=(pd.concat([allgoods,day2],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day3missing=(pd.concat([allgoods,day3],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day4missing=(pd.concat([allgoods,day4],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day5missing=(pd.concat([allgoods,day5],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day6missing=(pd.concat([allgoods,day6],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day7missing=(pd.concat([allgoods,day7],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day8missing=(pd.concat([allgoods,day8],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day9missing=(pd.concat([allgoods,day9],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day10missing=(pd.concat([allgoods,day10],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#day11missing=(pd.concat([allgoods,day11],sort=True).loc[:,"goods_id"]).to_frame().drop_duplicates(keep=False)
#
## here we have the goods that are not sold on some days. we put quantity = 0
## day1 and day4 not here, means every good was sold on those days
#day2missing.loc[:,"goods_amount"]=0
#day3missing.loc[:,"goods_amount"]=0
#day5missing.loc[:,"goods_amount"]=0
#day6missing.loc[:,"goods_amount"]=0
#day7missing.loc[:,"goods_amount"]=0
#day8missing.loc[:,"goods_amount"]=0
#day9missing.loc[:,"goods_amount"]=0
#day10missing.loc[:,"goods_amount"]=0
#day11missing.loc[:,"goods_amount"]=0
#
## combine with the list of goods sold for each day by concatenating 
#day2 = pd.concat([day2,day2missing])
#day2.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day2.reset_index(level=None, drop=True, inplace=True)
#day3 = pd.concat([day3,day3missing])
#day3.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day3.reset_index(level=None, drop=True, inplace=True)
#day5 = pd.concat([day5,day5missing])
#day5.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day5.reset_index(level=None, drop=True, inplace=True)
#day6 = pd.concat([day6,day6missing])
#day6.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day6.reset_index(level=None, drop=True, inplace=True)
#day7 = pd.concat([day7,day7missing])
#day7.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day7.reset_index(level=None, drop=True, inplace=True)
#day8 = pd.concat([day8,day8missing])
#day8.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day8.reset_index(level=None, drop=True, inplace=True)
#day9 = pd.concat([day9,day9missing])
#day9.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day9.reset_index(level=None, drop=True, inplace=True)
#day10 = pd.concat([day10,day10missing])
#day10.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day10.reset_index(level=None, drop=True, inplace=True)
#day11 = pd.concat([day11,day11missing])
#day11.sort_values(by='goods_id', axis=0, ascending=True, inplace=True)
#day11.reset_index(level=None, drop=True, inplace=True)

## get observations for each product
## numpy's std uses population variance(biased). statistics.stdev uses sample variance(unbiased). 
## to use sample variance, set ddof = 1 in np.std. use unbiased since sample size small? wrong, 
## calculate covariance matrix for multivar normal
#
## puts quantity ordered for each product in a list of lists, then convert to array
## each column is one product, each row is the quantity ordered for all products for a day
#product2=[list(day1.iloc[:,1]),list(day2.iloc[:,1]),list(day3.iloc[:,1]),list(day4.iloc[:,1]),list(day5.iloc[:,1]),list(day6.iloc[:,1]),list(day7.iloc[:,1]),list(day8.iloc[:,1]),list(day9.iloc[:,1]),list(day10.iloc[:,1]),list(day11.iloc[:,1])]
#product = np.array(product2)
#
## to obtain mle estimators for mean and cov
#productmean = np.mean(product,axis=0)
#productcov=0
#for i in range(11):
#    test1 = np.reshape((np.transpose(product[i,:])-productmean), (numvars,1))
#    test2 = np.reshape((np.transpose(product[i,:])-productmean), (1,numvars))
#    productcov += np.matmul(test1,test2)
#productcov = productcov/(i+1)
## divide by i+1 since MLE estimate of cov matrix is divide by N not N-1
#
##from scipy.stats import multivariate_normal
######################################## obtained parameters by mle for first price vector


# =============================================================================
# Generate market share X for each price vector
# =============================================================================
# where X is generated using a multinomial logit choice model
# X = exp(V(i)-P(i))/sum(exp(V(i)-P(i))) for i=0,1,...,N where i=0 is when customer buys nothing

# Number of arms
k = 3

# First we generate V. one V for all price vectors. V = +- 5% of Pbar
# Build V around the default price vector which is equivalent to the average of all 3 price vectors
v = np.empty([numvars,1])
np.random.seed(10)
for i in range(numvars):
    v[i] = np.random.randint(prices[1][i]-0.05*prices[1][i], prices[1][i]+0.05*prices[1][i]) # 1 since prices[1] is the default price vector

np.random.seed(13)
v0 = np.random.randint(0, 5)
# True demand: Theoretical X, where X is the demand
truedemand = [np.empty([numvars,1])] * k

# To store x0
x0 = np.zeros(k)

for arm in range(k):
    for i in range(numvars):
        truedemand[arm][i] = math.exp(v[i]-(prices[arm][i]))
    # x0 is the choice where they dont buy anything
    x0[arm] = math.exp(v0)/(sum(truedemand[arm])+math.exp(v0))
    truedemand[arm] = truedemand[arm]/(np.sum(truedemand[arm])+math.exp(v0))

# =============================================================================
# To generate additional data by creating epsilons for each price vector
# =============================================================================
# Number of epsilons. Multiply by 2 to get number of data points per arm. 
numep = 25

# To hold data for all arms
data_all = []

for arm in range(k):
    ep = np.empty([numvars,numep])
#    np.random.seed(10)
    for i in range(numvars):
        ep[i] = np.random.uniform(low=0.0, high=truedemand[arm][i]/5, size=numep)
    
    # create data via +epsilon and -epsilon so that the mean is still the theoretical mean
    temp1 = np.empty([numvars,numep])
    temp2 = np.empty([numvars,numep])
    for i in range(numvars):
        temp1[i] = truedemand[arm][i] + ep[i]
        temp2[i] = truedemand[arm][i] - ep[i]
    data_all.append(np.append(temp1, temp2, axis=1))
        

# Appending all data points together for dynamic_TS approach
dataall = np.append(data_all[0], data_all[1], axis=1)
for arm in range(1, k-1):
    dataall = np.append(dataall, data_all[arm+1], axis=1)

# Obtain revenue of each day, find SD to estimate sigma in dynamic_TS approach
revenue_of_prior = np.empty(numep*2*k) # each arm has numep*2 points. we have k arms.

# For each arm
for arm in range(k):
    # For each data point of each arm
    for i in range(numep*2):
        # must reshape because data_all[arm][:,i] is 1 dimension
        revenue_of_prior[i + (arm*(numep*2))] = np.sum(np.reshape(data_all[arm][:,i],(numvars,1)) * prices[arm]) 

# Estimate of variance
sighat = np.std(revenue_of_prior)

# Number of iterations
itr = 1000

# =============================================================================
# Thompson sampling done here (classical approach)
# =============================================================================
## Assume each price vector has normal distribution. Use MLE to estimate parameters from data created.
## Mean for each arm
#cTS_mean = [0.0]*k
#for arm in range(k):
#    cTS_mean[arm] = np.mean(data_all[arm], axis=1)
#
## Covariance
#cTS_cov = [0.0]*k
#for arm in range(k):
#    # Must reset cov = 0 for each arm other cumulative sum will affect later arms
#    cov = 0
#    for i in range(data_all[arm].shape[1]): # For each data point of each arm
#        temp1 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (numvars,1))
#        temp2 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (1,numvars))
#        cov += np.matmul(temp1, temp2)
#        
#    # divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
#    cTS_cov[arm] = cov/(numep*2)
#
## Initialise counter for the number of times each arm is selected
#cTS_counter = np.zeros(k)
#
## Revenue and basket
#revenue_cTS = 0
#basket_cTS = np.zeros(itr)
#
#"""Idea is to estimate the true underlying demand distribution using historical data.
#Using our estimated demand distribution, produce an estimate of the demand by sampling.
#Optimise/choose the price that will maximise revenue BASED ON the SAMPLED demand.
#OBSERVE the ACTUAL demand that comes from the true underlying demand distribution.
#Calculate the revenue that we obtain. Add this new information (price and demand)
#to our historical data and repeat.
#
#Since we dont have a distribution to sample from that represents the true demand, we shall replace that
#with the theoretical X since in the long run, random samples from the true demand distribution
#should be very close to the theoretical X.
#"""
#
##np.random.seed(10)
#for j in range(itr):
#    # Randomly sample from each distribution. This is our SAMPLED demand.
#    cTS_forecast_demand = [0.0]*k
#    for arm in range(k):
##        np.random.seed(10)
#        cTS_forecast_demand[arm] = np.random.multivariate_normal(cTS_mean[arm], cTS_cov[arm], 1).T
#    
#    # Calculate revenue based on the SAMPLED demand
#    cTS_forecast_revenue = np.zeros(k)
#    for arm in range(k):
#        cTS_forecast_revenue[arm] = np.sum(np.multiply(cTS_forecast_demand[arm], prices[arm]))
#       
#    # Choose the arm with the highest revenue based on SAMPLED demand.
#    arm = np.argmax(cTS_forecast_revenue)
#    
#    # Add to counter
#    cTS_counter[arm] += 1
#    
#    # Pull the arm, calculate and accumulate OBSERVED revenue
#    rev = np.sum(np.multiply(truedemand[arm], prices[arm]))
#    revenue_cTS += rev
#    basket_cTS[j] = rev
#    
#    # Adding observed/theoretical X to list of observations
#    data_all[arm] = np.append(data_all[arm], truedemand[arm], axis=1)
#    
#    # Recalculate parameters using MLE
#    # Mean for each arm
#    cTS_mean[arm] = np.mean(data_all[arm], axis=1)
#    
#    # Covariance
#    cov = 0
#    for i in range(data_all[arm].shape[1]): # For each data point of each arm
#        temp1 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (numvars,1))
#        temp2 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (1,numvars))
#        cov += np.matmul(temp1, temp2)       
#    # divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
#    cTS_cov[arm] = cov/(data_all[arm].shape[1])
#
## =============================================================================
## Validate if the chosen arm is correct (use theoretical X for each arm)
## =============================================================================
## 0:lower, 1:middle, 2:higher
## Real revenue and basket
#realrevenue = np.zeros(k)
#basket_real = np.zeros((itr, k))
#
#for arm in range(k):
#    for i in range(itr):
#        rev = np.sum(truedemand[arm] * prices[arm])
#        realrevenue[arm] += rev
#        basket_real[i][arm] = rev
#
#
## Graphical comparison of TS vs each arm
##plt.plot(np.cumsum(basket_cTS),'ko-')
##plt.plot(np.cumsum(basket_real[:,0]),'k--')
##plt.plot(np.cumsum(basket_real[:,1]),'k-.')
##plt.plot(np.cumsum(basket_real[:,2]),'k:')
##plt.ylabel('Cumulated revenue',fontsize=20)
##plt.xlabel('Time period',fontsize=20)
##plt.legend(['Real revenue','Lower arm','Middle arm','Higher arm'],fontsize=20)
##plt.title("Comparing results with all 3 arms", fontsize=20)
##plt.show()
#
## =============================================================================
## Upper confidence bound (UCB1 method, Hoeffding's inequality)
## =============================================================================
## Initialise counters
## 0:lower, 1:middle, 2:higher
#ucb_counter = np.ones(k)
#
## Initialise basket
#basket_ucb = np.zeros(itr)
#
## UCB scores
#ucb_scores = np.zeros(k)
#
## Initialise mean, pull each arm once
#ucb_mean = np.zeros(k)
#for arm in range(k):
#    rev = np.sum(truedemand[arm] * prices[arm])
#    ucb_mean[arm] = rev
#    basket_ucb[arm] = rev
#    
## Update ucb scores
#for arm in range(k):
#    ucb_scores[arm] = ucb_mean[arm] + sqrt(2*(log(k)/ucb_counter[arm]))
#
## Overall revenue
#revenue_ucb = np.sum(ucb_mean)
#
#for i in range(k, itr):
#    # Find the arm with highest ucb score
#    arm = np.argmax(ucb_scores)
#    
#    # Calculate revenue and add to basket
#    rev = np.sum(truedemand[arm] * prices[arm])
#    revenue_ucb += rev
#    basket_ucb[i] = rev
#    
#    # Increase counter and recalculate mean
#    ucb_counter[arm] += 1
#    ucb_mean[arm] += (rev - ucb_mean[arm])/(ucb_counter[arm])
#    
#    # Recalculate ucb score
#    ucb_scores[arm] = ucb_mean[arm] + sqrt(2*(log(i+1)/ucb_counter[arm])) # i+1 for number of iterations
#
## =============================================================================
## Epsilon-greedy algorithm
## =============================================================================
## Initialise counters
## 0:lower, 1:middle, 2:higher
#eg_counter = np.zeros(k)
#
## Initialise basket
#basket_eg = np.zeros(itr)
#
## Initialise mean
#eg_mean = np.zeros(k)
#    
## Overall revenue
#revenue_eg = 0  
#
## Set epsilon = 0.1
#e = 0.1
#
## Overall revenue for this algorithm
#revenue_eg = 0
#
#for i in range(itr):
#    ep = np.random.uniform()
#    
#    # Exploitation, run the best arm
#    if ep>e:
#        # Checking for the best arm
#        arm = np.argmax(eg_mean)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_eg += rev
#        basket_eg[i] = rev
#        
#        # Increase counter and recalculate mean
#        eg_counter[arm] += 1
#        eg_mean[arm] += (rev - eg_mean[arm])/(eg_counter[arm])
#                    
#    # Exploration, randomly select an arm
#    else:
#        arm = np.random.randint(0,3)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_eg += rev
#        basket_eg[i] = rev
#        
#        # Increase counter and recalculate mean
#        eg_counter[arm] += 1
#        eg_mean[arm] += (rev - eg_mean[arm])/(eg_counter[arm])
#
## =============================================================================
## Epsilon-greedy algorithm with optimistic initialisation
## =============================================================================
## Initialise counters
## 0:lower, 1:middle, 2:higher
#egoi_counter = np.zeros(k)
#
## Initialise basket
#basket_egoi = np.zeros(itr)
#
## Initialise mean for each arm as 100
#egoi_mean = [100.0]*k
#egoi_mean = np.array(egoi_mean)
#    
## Overall revenue
#revenue_egoi = 0  
#
## Set epsilon = 0.1
#e = 0.1
#
## Overall revenue for this algorithm
#revenue_egoi = 0
#
#for i in range(itr):
#    ep = np.random.uniform()
#    
#    # Exploitation, run the best arm
#    if ep>e:
#        # Checking for the best arm
#        arm = np.argmax(egoi_mean)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_egoi += rev
#        basket_egoi[i] = rev
#        
#        # Increase counter and recalculate mean
#        egoi_counter[arm] += 1
#        egoi_mean[arm] += (rev - egoi_mean[arm])/(egoi_counter[arm])
#                    
#    # Exploration, randomly select an arm
#    else:
#        arm = np.random.randint(0,3)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_egoi += rev
#        basket_egoi[i] = rev
#        
#        # Increase counter and recalculate mean
#        egoi_counter[arm] += 1
#        egoi_mean[arm] += (rev - egoi_mean[arm])/(egoi_counter[arm])
#        
#    
## =============================================================================
## Epsilon-greedy algorithm with decay
## =============================================================================
## Initialise counters
## 0:lower, 1:middle, 2:higher
#egd_counter = np.zeros(k)
#
## Initialise mean to be 100 for each arm
#egd_mean = [100.0]*k
#egd_mean = np.array(egd_mean)
#
## Set epsilon using this formula
## e = 1/(1+n/k) where k is the number of arms and n is the number of iterations already passed
#
## Overall revenue for this algorithm
#revenue_egd = 0
#basket_egd = np.zeros(itr)
#
#for i in range(itr):
#    ep = np.random.uniform()
#    
#    # Exploitation, run the best arm
#    if ep > 1/(1+i/k):
#        # Checking for the best arm
#        arm = np.argmax(egd_mean)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_egd += rev
#        egd_counter[arm] += 1
#        basket_egd[i] = rev
#        
#        # Recalculate mean
#        egd_mean[arm] += (rev - egd_mean[arm])/(egd_counter[arm] + 1) # Since we have initialisation =/= 0
# 
#    # Exploration, randomly select an arm
#    else:
#        arm = np.random.randint(0,k)
#        
#        # Calculate revenue and add to basket
#        rev = np.sum(truedemand[arm] * prices[arm])
#        revenue_egd += rev
#        egd_counter[arm] += 1
#        basket_egd[i] = rev
#        
#        # Recalculate mean
#        egd_mean[arm] += (rev - egd_mean[arm])/(egd_counter[arm] + 1) # Since we have initialisation =/= 0
#        
## =============================================================================
## UCB1-Tuned
## =============================================================================
## Initialise counters
## 0:lower, 1:middle, 2:higher
#ucbt_counter = np.ones(k)
#
## Initialise basket
#basket_ucbt = np.zeros(itr)
#
## UCB scores
#ucbt_scores = np.zeros(k)
#
## sum of squares for ucb1-tuned
#ucbt_ss = np.zeros(k)
#
## Initialise mean, pull each arm once
#ucbt_mean = np.zeros(k)
#for arm in range(k):
#    rev = np.sum(truedemand[arm] * prices[arm])
#    ucbt_mean[arm] = rev
#    basket_ucbt[arm] = rev
#    ucbt_ss[arm] += rev**2
#    
## Update ucb scores
#for arm in range(k):
#    # I think the calculation of scores here is wrong? ss should be variance of that arm.
#    ucbt_scores[arm] = ucbt_mean[arm] + sqrt((log(k)/ucbt_counter[arm]) * min(1/4, (ucbt_ss[arm] + 2*(log(k)/ucbt_counter[arm]))) )
#
## Overall revenue
#revenue_ucbt = np.sum(ucbt_mean)
#
#for i in range(k, itr):
#    # Find the arm with highest ucb score
#    arm = np.argmax(ucbt_scores)
#    
#    # Calculate revenue and add to basket
#    rev = np.sum(truedemand[arm] * prices[arm])
#    revenue_ucbt += rev
#    basket_ucbt[i] = rev
#    
#    # Increase counter and recalculate mean, sum of squares
#    ucbt_counter[arm] += 1
#    ucbt_mean[arm] += (rev - ucbt_mean[arm])/(ucbt_counter[arm])
#    ucbt_ss[arm] += rev**2
#    
#    # Recalculate ucb score
#    ucbt_scores[arm] = ucbt_mean[arm] + sqrt((log(i+1)/ucbt_counter[arm]) * min(1/4, (ucbt_ss[arm] + 2*(log(i+1)/ucbt_counter[arm]))) ) # i+1 for number of iterations
#           
## Graphical comparison of results
#plt.plot(np.cumsum(basket_cTS),'ko-')
#plt.plot(np.cumsum(basket_eg),'k--')
#plt.plot(np.cumsum(basket_egd),'k-.')
#plt.plot(np.cumsum(basket_egoi),'k:')
#plt.plot(np.cumsum(basket_ucb),'k+-.')
#plt.plot(np.cumsum(basket_ucbt),'kx:')
#plt.ylabel('Cumulated revenue',fontsize=20)
#plt.xlabel('Time period',fontsize=20)
#plt.legend(['classical Thompson sampling','epsilon-greedy','epsilon-greedy w/ decay','epsilon-greedy w/ optimistic initialisation','ucb1','ucb1-tuned'],fontsize=20)
#plt.title("Comparing TS with other algorithms",fontsize=20)
#plt.show()
#
#
#
# =============================================================================
# Thomson sampling (Dynamic pricing approach)
# =============================================================================




# Initialising prior distribution

# PED depends on the sequence of price vector used. Assume low-normal-high is the sequence. 
# normal - low
temp1 = (((truedemand[1]-truedemand[0])/truedemand[0])/((prices[1]-prices[0])/prices[0]))
# high - normal
temp2 = (((truedemand[2]-truedemand[1])/truedemand[1])/((prices[2]-prices[1])/prices[1]))

# Use elasticity estimates as mean and cov of prior distribution
elastmean = (temp1+temp2)/2
# constant c
c = 0.1 * np.mean(elastmean)
elastmean = np.reshape(np.array(elastmean),(numvars,1))
elastcov = c*np.identity(numvars)

# Preparing data for var model, need to insert date-time 
dataall = np.transpose(dataall)
dataall = pd.DataFrame(dataall)
dataall.insert(0, 'date_time', list(range(numep*6)))
dataall.iloc[:,0] = pd.to_datetime(dataall.iloc[:,0]) # converting date_time column to datetime64[ns] data type
dataall.index = dataall.date_time # change index to datetime since index must be datetime to run time series models
dataall = dataall.drop(['date_time'], axis=1) # remove datetime from column
#dataall.dtypes # to check if date_time is datetime64[ns]

# Making data stationary
#datadiff = np.log(dataall).diff().dropna() # diff(log)
#datadiff = dataall.diff().dropna() # diff()
#datadiff = dataall # without diff and log

datadifforg = dataall.diff().dropna() # log(abs(diff))
datadiff = np.log(abs(datadifforg))

# Time series model for forecasting
from statsmodels.tsa.api import VAR
# Run a VAR model on DIFFERENCED data
model = VAR(datadiff) 
# Estimate the coefficients of the VAR model
# select_order should show the information criterion for each number of lag
#model.select_order() # >1 lag then matrix is not positive definite. w/o diff and log, cant even do lag 1.
results = model.fit() 
# lag_order is the best lag chosen by model.fit()
lag_order = results.k_ar

prevprice = productpricehigher.iloc[:,1]
prevprice = np.array(prevprice)
prevprice = np.reshape(prevprice, (numvars,1))

realrevenue_dynamic_TS = 0

#f = results.forecast(datadiff.values[-lag_order:], 1) # with diff and log
#f = np.reshape(f, (numvars,1))
#f = np.e**(f + np.log(np.reshape(np.array(dataall.iloc[149,:]), (numvars,1))))

#f = results.forecast(datadiff.values[-lag_order:], 1) # with abs(diff) THEN log
#f = np.reshape(f, (numvars,1))
#f = np.e**(f) + np.reshape(np.array(dataall.iloc[149,:]), (numvars,1))

# testing for cointegration
#from statsmodels.tsa.vector_ar.vecm import coint_johansen
#haha = dataall.iloc[:,range(60,66)]
#coint_johansen(haha,-1,1).eig

import cplex
from scipy.optimize import minimize
from scipy.optimize import Bounds
# Set bounds to be lower prices and higher prices
bounds = Bounds(prevprice*0.9, prevprice*1.1)
from scipy.optimize import differential_evolution
import sys, os, mosek






for j in range(10):
    print(j)
    # Random sample for elasticities
    np.random.seed(10)
    elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
    # Ensures that all components are negative 
    np.random.seed(10)
    while (elast<0).all() == False: 
        print("elast is positive")
        print(j)
        elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
    elast = np.reshape(elast, (numvars,1))
    print("random sample ok")
    
    # Obtain demand forecast f and reshape
    """without diff and log"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (numvars,1))
    """diff()"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (numvars,1))
#    f = f + np.reshape(np.array(dataall.iloc[-lag_order,:]), (numvars,1)) 
    """diff(log)"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (numvars,1))
#    f = np.e**(f + np.log(np.reshape(np.array(dataall.iloc[-lag_order,:]), (numvars,1))))  
    """log(abs(diff))"""
    f = results.forecast(datadiff.values[-lag_order:], 1)
    f = np.reshape(f, (numvars,1))
    f = np.e**(f) + np.reshape(np.array(dataall.iloc[-lag_order,:]), (numvars,1))

    # Checking for valid f
    if (f>=0).all() == False:
        print("f is negative")
        break
    print("demand forecast ok")

    """using mosek"""
#    # Since the actual value of Infinity is ignored, we define it solely for symbolic purposes:
#    inf = 0.0
#
#    # Define a stream printer to grab output from MOSEK
#    def streamprinter(text):
#        sys.stdout.write(text)
#        sys.stdout.flush()
#    
#    def main():
#    # Open MOSEK and create an environment and task
#    # Make a MOSEK environment
#        with mosek.Env() as env:
#            # Attach a printer to the environment
#            env.set_Stream(mosek.streamtype.log, streamprinter)
#            # Create a task
#            with env.Task() as task:
#                task.set_Stream(mosek.streamtype.log, streamprinter)
#                
#                # Bound keys for variables
#                numvar = numvars
#                bkx = [mosek.boundkey.ra] * numvar
#                
#                # Bound values for variables
#                temppricelow = prevprice*0.9
#                temppricehigh = prevprice*1.1
#                blx = []
#                bux = []
#                for i in range(numvar):
#                    blx.append(temppricelow[i][0])
#                    bux.append(temppricehigh[i][0])
#                
#                # Objective linear coefficients
#                temp = f - (f * elast)
#                c = []
#                for i in range(numvar):
#                    c.append(temp[i][0])
##                print(c)
#                
#                # Append 'numcon' empty constraints.
#                # The constraints will initially have no bounds.
#                task.appendcons(0)
#            
#                # Append 'numvar' variables.
#                # The variables will initially be fixed at zero (x=0).
#                task.appendvars(numvar)
#    
#                for j in range(numvar):
#                    # Set the linear term c_j in the objective.
#                    task.putcj(j, c[j])
#                    
#                    # Set the bounds on variable j
#                    # blx[j] <= x_j <= bux[j] 
#                    task.putvarbound(j, bkx[j], blx[j], bux[j]) 
#
#                # Set up and input quadratic objective
#                qsubi = []
#                for i in range(numvar):
#                    qsubi.append(i)
#                qsubj = qsubi
#                temp = 2 * f * elast / prevprice # Must remember to *2, see mosek documentation
#                qval = []
#                for i in range(numvar):
#                    qval.append(temp[i][0])
##                print(qval)
#    
#                task.putqobj(qsubi, qsubj, qval)
#    
#                # Input the objective sense (minimize/maximize)
#                task.putobjsense(mosek.objsense.maximize)
#                task.analyzeproblem(mosek.streamtype.log)
#    
#                # Optimize
#                task.optimize()
#                
#                # Print a summary containing information
#                # about the solution for debugging purposes
##                task.solutionsummary(mosek.streamtype.msg)
#    
#                # Output a solution
#                xx = [0.] * numvar
#                task.getxx(mosek.soltype.itr, xx)
#                
#                print("===== Check here =====")
#                # To return quadratic coefficients
#                qtemp = np.empty([numvar,numvar])
#                for row in range(numvar):
#                    for col in range(numvar):
#                        qtemp[row][col] = task.getqobjij(row,col)
#                        
#                print("no. of constraints =",task.getnumcon())
#                print("no. of nonzero elements in quadratic objective terms =",task.getnumqobjnz())
#                print("no. of cones =",task.getnumcone())
#                print("no. of variables =",task.getnumvar())
#                print("Objective sense =",task.getobjsense())
#                print("Problem type =",task.getprobtype())
#                print("Problem status =",task.getprosta(mosek.soltype.itr)) # feasible
#                
#                # Get variable bounds
#                varbound = []
#                for b in range(numvar):
#                    varbound.append(task.getvarbound(b))
#                print("===== End of check =====")
#                
#                # Linear constraint
##                for index in range(numvar):
##                    print("no. of nonzero elements in {}-th column of A = {}".format(index,task.getacolnumnz(index)))
#                      
#                # To return linear coefficients
#                lineartemp = np.empty([numvar,1])
#                for index in range(numvar):
#                    lineartemp[index] = task.getcj(index)
#                    
#                return (xx, qtemp, lineartemp, varbound)
#            
#    # call the main function
#    result_mosek = main()
#    linear_coeff = result_mosek[2]
#    linear_check = f - (f * elast)
#    quad_coeff = result_mosek[1]
#    quad_check = 2 * f * elast / prevprice
#    var_bounds = result_mosek[3]
#    
#    newprice = result_mosek[0]
#    newprice = np.array(newprice)
#    newprice = np.reshape(newprice, (numvars,1))
    
    """using scipy.optimize"""   
    # Objective function, multiply by -1 since we want to maximize
    def eqn7(p):
        return -1.0*np.sum(p*p*f.flatten()*elast.flatten()/prevprice.flatten() - p*f.flatten()*elast.flatten() + p*f.flatten())
    
    # Initial guess is 1.05 * previous price
    bounds = Bounds(prevprice.flatten()*0.9, prevprice.flatten()*1.1)
    opresult = minimize(eqn7, prevprice.flatten()*1.05, bounds=bounds)
    newprice = opresult.x
    newprice = np.reshape(newprice, (numvars,1))  
    
    """using cplex"""
#    # create an instance
#    problem = cplex.Cplex()
#    
#    # set the function to maximise instead of minimise
#    problem.objective.set_sense(problem.objective.sense.maximize)
#    
#    # Adds variables
#    indices = problem.variables.add(names = [str(i) for i in range(numvars)])
#    
#    # Changes the linear part of the objective function.
#    for i in range(numvars):
#        problem.objective.set_linear(i, float(f[i]-f[i]*elast[i])) # form is objective.set_linear(var, value)
#        
#    # Sets the quadratic part of the objective function.
#    quad = (f*elast/prevprice)
#    problem.objective.set_quadratic([float(i) for i in quad])
#    
#    # Sets the lower bound for a variable or set of variables
#    for i in range(numvars):
#        problem.variables.set_lower_bounds(i, prevprice[i][0]*0.9)
#    
#    # Sets the upper bound for a variable or set of variables
#    for i in range(numvars):
#        problem.variables.set_upper_bounds(i, prevprice[i][0]*1.1)
#    
#    problem.solve()
#    newprice=problem.solution.get_values()
#    newprice = np.array(newprice)
    print("optimization ok")
    
    
    
    
    # Apply newprice to obtain observed demand
    observedx = np.empty([numvars,1])
    for i in range(numvars):
        observedx[i] = math.exp(v[i]-newprice[i])
    observedx = observedx/(np.sum(observedx)+math.exp(v0))
    print("observed demand ok")
    
    # Accumulate revenue
    realrevenue_dynamic_TS += np.sum(np.multiply(observedx, np.reshape(newprice,(numvars,1))))
    
    # Add observed demand to observations
    observedx = pd.DataFrame(observedx)
    dataall = dataall.append(observedx.transpose())
    dataall.insert(0, 'date_time', list(range(numep*6+j+1)))
    dataall.iloc[:,0] = pd.to_datetime(dataall.iloc[:,0]) # converting date_time column to datetime data type
    dataall.index = dataall.date_time # change index to datetime
    dataall = dataall.drop(['date_time'], axis=1) # remove datetime from column
    
#    datadiff = np.log(dataall).diff().dropna() # diff(log)
#    datadiff = dataall.diff().dropna() # diff()
#    datadiff = dataall # without diff and log

    datadifforg = dataall.diff().dropna() # log(abs(diff))
    datadiff = np.log(abs(datadifforg))
    print("add demand ok")
    
    # Re-estimate VAR model
    model = VAR(datadiff)
    results = model.fit()
    lag_order = results.k_ar
    print("re-estimate var model ok")
    
    # For M inverse matrix
    thet = np.multiply(np.reshape(newprice**2,(numvars,1)),f)
    thet = np.divide(thet, prevprice)
    thet = thet - np.multiply(np.reshape(newprice,(numvars,1)),f)
    minv = (thet*np.transpose(thet))/sighat**2 + 1e-5*np.identity(numvars) # fix lambda = 1e-5
    print("minv ok")
    
    # For M inverse beta matrix
    rbar = np.sum(np.multiply(np.reshape(newprice,(numvars,1)),f))
    rt = float(np.sum(np.multiply(observedx, np.reshape(newprice,(numvars,1)))))
    minvb = (rt - rbar)/(sighat**2)*thet
    print("minvb ok")
    
    # Update mean of elasticity's distribution
    pt1 = np.linalg.inv(np.linalg.inv(elastcov) + minv)
    pt2 = np.matmul(np.linalg.inv(elastcov), elastmean) + minvb
    elastmean = np.matmul(pt1, pt2)
    print("update mean ok")
    
    # Update covariance of elasticity's distribution
    elastcov = pt1
    
    # Update prevprice to new price
    prevprice = np.reshape(newprice, (numvars,1))
    








