# -*- coding: utf-8 -*-

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


# =============================================================================
# checking for full-cut pricing
# =============================================================================
##%matplotlib qt
#
#mylist=[]
## histogram using 99 as threshold 
###datamerge.hist(column="goods_money",bins=[0,99,99*2,99*3,99*4,99*5,99*6,99*7])
#for i in range(1,8): 
#    # calculate mean of full_cut for each range 
#    mylist.append(datamerge.loc[(datamerge["goods_money"]>=99*(i-1)) & (datamerge["goods_money"]<99*i),:].mean()[2])
#
#mylist2=[]    
## histogram using 80 as threshold 
###datamerge.hist(column="goods_money",bins=[0,80,80*2,80*3,80*4,80*5,80*6,80*7,80*8,80*9])
#for i in range(1,8): 
#    # calculate mean of full_cut for each range 
#    mylist2.append(datamerge.loc[(datamerge["goods_money"]>=80*(i-1)) & (datamerge["goods_money"]<80*i),:].mean()[2])
#
#mylist3=[]
## histogram using 120 as threshold 
###datamerge.hist(column="goods_money",bins=[0,120,120*2,120*3,120*4,120*5,120*6,120*7])
#for i in range(1,8): 
#    # calculate mean of full_cut for each range 
#    mylist3.append(datamerge.loc[(datamerge["goods_money"]>=120*(i-1)) & (datamerge["goods_money"]<120*i),:].mean()[2])
#
#mylist4=[]
## histogram using 110 as threshold 
###datamerge.hist(column="goods_money",bins=[0,110,110*2,110*3,110*4,110*5,110*6,110*7])
#for i in range(1,8): 
#    # calculate mean of full_cut for each range 
#    mylist4.append(datamerge.loc[(datamerge["goods_money"]>=110*(i-1)) & (datamerge["goods_money"]<110*i),:].mean()[2])
#
#mylist5=[]
## histogram using 90 as threshold 
###datamerge.hist(column="goods_money",bins=[0,90,90*2,90*3,90*4,90*5,90*6,90*7,90*8])
#for i in range(1,8): 
#    # calculate mean of full_cut for each range 
#    mylist5.append(datamerge.loc[(datamerge["goods_money"]>=90*(i-1)) & (datamerge["goods_money"]<90*i),:].mean()[2])
#
#plt.plot(mylist2,'r', mylist5,'b', mylist,'g', mylist4,'m', mylist3, 'y')
#plt.ylabel('Mean discount',fontsize=15)
#plt.xlabel('Groups',fontsize=15)
#plt.legend(['X = 80','X = 90','X = 99','X = 110', 'X = 120'],fontsize=20)
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

###########################################################################################
########################################## now trying to use mle for first price vector
# 11 days, take as 11 observations for each good
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
#
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

# First we generate V. one V for all price vectors. V = +- 5% of Pbar
# Build V around the default price vector which is equivalent to the average of all 3 price vectors
v = np.empty([numvars,1])
#np.random.seed(10)
for i in range(numvars):
    v[i] = np.random.randint(productprice.iloc[i,1]-0.05*productprice.iloc[i,1], productprice.iloc[i,1]+0.05*productprice.iloc[i,1])
#np.random.seed(10)
v0=np.random.randint(0, 5)
"""np.random.randint got size arg. can we use array operations instead of looping?"""
# Theoretical X, where X is the demand
xprice = np.empty([numvars,1])
xpricehigher = np.empty([numvars,1])
xpricelower = np.empty([numvars,1])

for i in range(numvars):
    xprice[i] = math.exp(v[i]-(productprice.iloc[:,1])[i])
# x0 is the choice where they dont buy anything
x0 = math.exp(v0)/(sum(xprice)+math.exp(v0))
xprice = xprice/(np.sum(xprice)+math.exp(v0))

#jaja = np.empty([numvars,1])
#jaja = math.exp(v-productprice.iloc[:,1])
#jaja = jaja/(np.sum(jaja)+math.exp(v0))



for i in range(numvars):
    xpricehigher[i] = math.exp(v[i]-(productpricehigher.iloc[:,1])[i])
x0higher = math.exp(v0)/(sum(xpricehigher)+math.exp(v0))
xpricehigher = xpricehigher/(np.sum(xpricehigher)+math.exp(v0))

for i in range(numvars):
    xpricelower[i] = math.exp(v[i]-(productpricelower.iloc[:,1])[i])
x0lower = math.exp(v0)/(sum(xpricelower)+math.exp(v0))
xpricelower = xpricelower/(np.sum(xpricelower)+math.exp(v0))

# =============================================================================
# To generate additional data by creating epsilons for each price vector
# =============================================================================
# Number of epsilons. Multiply by 2 to get number of data points per arm. 
numep = 25
# First price vector
ep1 = np.empty([numvars,numep])
#np.random.seed(10)
for i in range(numvars):
    ep1[i] = np.random.uniform(low=0.0, high=xprice[i]/5, size=numep)

# create data via +epsilon and -epsilon so that the mean is still the theoretical mean
temp1 = np.empty([numvars,numep])
temp2 = np.empty([numvars,numep])
for i in range(numvars):
    temp1[i] = xprice[i]+ep1[i]
    temp2[i] = xprice[i]-ep1[i]
datax = np.append(temp1, temp2, axis=1)

# Second price vector
ep2 = np.empty([numvars,numep])
#np.random.seed(10)
for i in range(numvars):
    ep2[i] = np.random.uniform(low=0.0, high=xpricelower[i]/5, size=numep)

temp1 = np.empty([numvars,numep])
temp2 = np.empty([numvars,numep])
for i in range(numvars):
    temp1[i] = xpricelower[i]+ep2[i]
    temp2[i] = xpricelower[i]-ep2[i]
dataxlower = np.append(temp1, temp2, axis=1)

# Third price vector
ep3 = np.empty([numvars,numep])
#np.random.seed(10)
for i in range(numvars):
    ep3[i] = np.random.uniform(low=0.0, high=xpricehigher[i]/5, size=numep)

temp1 = np.empty([numvars,numep])
temp2 = np.empty([numvars,numep])
for i in range(numvars):
    temp1[i] = xpricehigher[i]+ep3[i]
    temp2[i] = xpricehigher[i]-ep3[i]
dataxhigher=np.append(temp1, temp2, axis=1)

# Appending all data points together for dynamic_TS approach
dataall = np.append(dataxlower, datax, axis=1)
dataall = np.append(dataall, dataxhigher, axis=1)

# Obtain revenue of each day, find SD to estimate sigma in dynamic_TS approach
rev1 = np.empty(numep*2)
for i in range(numep*2):
    rev1[i] = sum(np.multiply(dataxlower[:,i], productpricelower.iloc[:,1]))
rev2 = np.empty(numep*2)
for i in range(numep*2):
    rev2[i] = sum(np.multiply(datax[:,i], productprice.iloc[:,1]))
rev3 = np.empty(numep*2)
for i in range(numep*2):
    rev3[i] = sum(np.multiply(dataxhigher[:,i], productpricehigher.iloc[:,1]))

revall = np.append(rev1, rev2)
revall = np.append(revall, rev3)
sighat = np.std(revall)

# Number of iterations
itr = 1000
# Number of arms
k = 3
# True demand
truedemand = np.array([xpricelower,xprice,xpricehigher])
# Prices for each arm
prices = np.array([productpricelower.iloc[:,1].values.reshape((numvars,1)), productprice.iloc[:,1].values.reshape((numvars,1)), productpricehigher.iloc[:,1].values.reshape((numvars,1))])

# =============================================================================
# Thompson sampling done here (classical approach)
# =============================================================================
# Assume each price vector has normal distribution. Use MLE to estimate parameters from data created.
# First price vector
productmean = np.mean(datax, axis=1)
productcov = 0
for i in range(datax.shape[1]):
    test1 = np.reshape((datax[:,i]-productmean), (numvars,1))
    test2 = np.reshape((datax[:,i]-productmean), (1,numvars))
    productcov += np.matmul(test1, test2)
productcov = productcov/(i+1)
# divide by i+1 since MLE estimate of cov matrix is divide by N not N-1

# Second price vector
productmeanlower = np.mean(dataxlower, axis=1)
productcovlower = 0
for i in range(dataxlower.shape[1]):
    test1 = np.reshape((dataxlower[:,i]-productmeanlower), (numvars,1))
    test2 = np.reshape((dataxlower[:,i]-productmeanlower), (1,numvars))
    productcovlower += np.matmul(test1, test2)
productcovlower = productcovlower/(i+1)

# Third price vector
productmeanhigher = np.mean(dataxhigher, axis=1)
productcovhigher = 0
for i in range(dataxhigher.shape[1]):
    test1 = np.reshape((dataxhigher[:,i]-productmeanhigher), (numvars,1))
    test2 = np.reshape((dataxhigher[:,i]-productmeanhigher), (1,numvars))
    productcovhigher += np.matmul(test1, test2)
productcovhigher = productcovhigher/(i+1)

# Initialise counter for the number of times each arm is selected
cTS_counter_middle = 0
cTS_counter_lower = 0
cTS_counter_higher = 0
revenue_cTS = 0
basket_cTS = np.zeros(itr)

"""Idea is to estimate the true underlying demand distribution using historical data.
Using our estimated demand distribution, produce an estimate of the demand by sampling.
Optimise/choose the price that will maximise revenue BASED ON the SAMPLED demand.
OBSERVE the ACTUAL demand that comes from the true underlying demand distribution.
Calculate the revenue that we obtain. Add this new information (price and demand)
to our historical data and repeat.

Since we dont have a distribution to sample from that represents the true demand, we shall replace that
with the theoretical X since in the long run, random samples from the true demand distribution
should be very close to the theoretical X.
"""
mid = productcov
low = productcovlower
high = productcovhigher

#np.random.seed(10)
for j in range(itr):
    # Randomly sample from each distribution. This is our SAMPLED demand.
    forecastdemand_middle = np.random.multivariate_normal(productmean, productcov,1).T
    forecastdemand_lower = np.random.multivariate_normal(productmeanlower, productcovlower,1).T
    forecastdemand_higher = np.random.multivariate_normal(productmeanhigher, productcovhigher,1).T
    
    # Calculate revenue based on the SAMPLED demand
    forecastrevenue_middle = np.multiply(forecastdemand_middle, productprice.iloc[:,1].values.reshape((numvars,1)))
    forecastrevenue_lower = np.multiply(forecastdemand_lower, productpricelower.iloc[:,1].values.reshape((numvars,1)))
    forecastrevenue_higher = np.multiply(forecastdemand_higher, productpricehigher.iloc[:,1].values.reshape((numvars,1)))
    
    # Choose the arm with the highest revenue based on SAMPLED demand.
    # Middle arm is best
    if np.sum(forecastrevenue_middle)>np.sum(forecastrevenue_higher) and np.sum(forecastrevenue_middle)>np.sum(forecastrevenue_lower):
        cTS_counter_middle += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
#        truerdm = np.random.multivariate_normal(xprice.flatten(),mid,1).T
#        realrevenue_classical_TS += np.sum(np.multiply(truerdm,productprice.iloc[:,1].values.reshape((numvars,1))))
        rev = np.sum(np.multiply(xprice, productprice.iloc[:,1].values.reshape((numvars,1))))
        revenue_cTS += rev
        basket_cTS[j] = rev
        
        # Adding observed/theoretical X to list of observations
#        datax = np.append(datax, truerdm, axis=1)
        datax = np.append(datax, xprice, axis=1)
      
        # Recalculate parameters using MLE
        productmean = np.mean(datax, axis=1)
        productcov = 0
        for i in range(datax.shape[1]):
            test1 = np.reshape((datax[:,i]-productmean), (numvars,1))
            test2 = np.reshape((datax[:,i]-productmean), (1,numvars))
            productcov += np.matmul(test1, test2)
        productcov = productcov/(i+1)

    # Lower arm is best
    elif np.sum(forecastrevenue_lower)>np.sum(forecastrevenue_middle) and np.sum(forecastrevenue_lower)>np.sum(forecastrevenue_higher):
        cTS_counter_lower += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
#        truerdmlower = np.random.multivariate_normal(xpricelower.flatten(),low,1).T
#        realrevenue_classical_TS += np.sum(np.multiply(truerdmlower,productpricelower.iloc[:,1].values.reshape((numvars,1))))
        rev = np.sum(np.multiply(xpricelower, productpricelower.iloc[:,1].values.reshape((numvars,1))))
        revenue_cTS += rev
        basket_cTS[j] = rev
        
        # Adding observed/theoretical X to list of observations
#        dataxlower = np.append(dataxlower, truerdmlower, axis=1)
        dataxlower = np.append(dataxlower, xpricelower, axis=1)
        
        # Recalculate parameters using MLE
        productmeanlower = np.mean(dataxlower, axis=1)
        productcovlower = 0
        for i in range(dataxlower.shape[1]):
            test1 = np.reshape((dataxlower[:,i]-productmeanlower), (numvars,1))
            test2 = np.reshape((dataxlower[:,i]-productmeanlower), (1,numvars))
            productcovlower += np.matmul(test1, test2)
        productcovlower = productcovlower/(i+1)

    # Higher arm is best
    else:
        cTS_counter_higher += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
#        truerdmhigher = np.random.multivariate_normal(xpricehigher.flatten(),low,1).T
#        realrevenue_classical_TS += np.sum(np.multiply(truerdmhigher,productpricehigher.iloc[:,1].values.reshape((numvars,1))))
        rev = np.sum(np.multiply(xpricehigher, productpricehigher.iloc[:,1].values.reshape((numvars,1))))
        revenue_cTS += rev
        basket_cTS[j] = rev
        
        # Adding observed/theoretical X to list of observations
#        dataxhigher = np.append(dataxhigher, truerdmhigher, axis=1)
        dataxhigher = np.append(dataxhigher, xpricehigher, axis=1)
        
        # Recalculate parameters using MLE
        productmeanhigher = np.mean(dataxhigher, axis=1)
        productcovhigher = 0
        for i in range(dataxhigher.shape[1]):
            test1 = np.reshape((dataxhigher[:,i]-productmeanhigher), (numvars,1))
            test2 = np.reshape((dataxhigher[:,i]-productmeanhigher), (1,numvars))
            productcovhigher += np.matmul(test1, test2)
        productcovhigher = productcovhigher/(i+1)

# =============================================================================
# Validate if the chosen arm is correct (use theoretical X for each arm)
# =============================================================================
# 0:lower, 1:middle, 2:higher
# Real revenue and basket
realrevenue = np.zeros(k)
basket_real = np.zeros((itr, k))

for arm in range(k):
    for i in range(itr):
        rev = np.sum(truedemand[arm] * prices[arm])
        realrevenue[arm] += rev
        basket_real[i][arm] = rev


# Graphical comparison of TS vs each arm
#plt.plot(np.cumsum(basket_cTS),'r')
#plt.plot(np.cumsum(basket_real[:,0]),'b')
#plt.plot(np.cumsum(basket_real[:,1]),'y')
#plt.plot(np.cumsum(basket_real[:,2]),'m')
#plt.ylabel('Cumulated revenue',fontsize=15)
#plt.xlabel('Time period',fontsize=15)
#plt.legend(['Real revenue','Lower arm','Middle arm','Higher arm'],fontsize=20)
#plt.show()


# =============================================================================
# Upper confidence bound (UCB1 method, Hoeffding's inequality)
# =============================================================================
# Initialise counters
# 0:lower, 1:middle, 2:higher
ucb_counter = np.ones(k)

# Initialise basket
basket_ucb = np.zeros(itr)

# UCB scores
ucb_scores = np.zeros(k)

# Initialise mean, pull each arm once
ucb_mean = np.zeros(k)
for arm in range(k):
    rev = np.sum(truedemand[arm] * prices[arm])
    ucb_mean[arm] = rev
    basket_ucb[arm] = rev
    
# Update ucb scores
for arm in range(k):
    ucb_scores[arm] = ucb_mean[arm] + sqrt(2*(log(k)/ucb_counter[arm]))

# Overall revenue
revenue_ucb = np.sum(ucb_mean)

for i in range(k, itr):
    # Find the arm with highest ucb score
    arm = np.argmax(ucb_scores)
    
    # Calculate revenue and add to basket
    rev = np.sum(truedemand[arm] * prices[arm])
    revenue_ucb += rev
    basket_ucb[i] = rev
    
    # Increase counter and recalculate mean
    ucb_counter[arm] += 1
    ucb_mean[arm] += (rev - ucb_mean[arm])/(ucb_counter[arm])
    
    # Recalculate ucb score
    ucb_scores[arm] = ucb_mean[arm] + sqrt(2*(log(i+1)/ucb_counter[arm])) # i+1 for number of iterations

# =============================================================================
# Epsilon-greedy algorithm
# =============================================================================
# Initialise counters
# 0:lower, 1:middle, 2:higher
eg_counter = np.zeros(k)

# Initialise basket
basket_eg = np.zeros(itr)

# Initialise mean
eg_mean = np.zeros(k)
    
# Overall revenue
revenue_eg = 0  

# Set epsilon = 0.1
e = 0.1

# Overall revenue for this algorithm
revenue_eg = 0

for i in range(itr):
    ep = np.random.uniform()
    
    # Exploitation, run the best arm
    if ep>e:
        # Checking for the best arm
        arm = np.argmax(eg_mean)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_eg += rev
        basket_eg[i] = rev
        
        # Increase counter and recalculate mean
        eg_counter[arm] += 1
        eg_mean[arm] += (rev - eg_mean[arm])/(eg_counter[arm])
                    
    # Exploration, randomly select an arm
    else:
        arm = np.random.randint(0,3)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_eg += rev
        basket_eg[i] = rev
        
        # Increase counter and recalculate mean
        eg_counter[arm] += 1
        eg_mean[arm] += (rev - eg_mean[arm])/(eg_counter[arm])

# =============================================================================
# Epsilon-greedy algorithm with optimistic initialisation
# =============================================================================
# Initialise counters
# 0:lower, 1:middle, 2:higher
egoi_counter = np.zeros(k)

# Initialise basket
basket_egoi = np.zeros(itr)

# Initialise mean for each arm as 100
egoi_mean = [100.0]*k
egoi_mean = np.array(egoi_mean)
    
# Overall revenue
revenue_egoi = 0  

# Set epsilon = 0.1
e = 0.1

# Overall revenue for this algorithm
revenue_egoi = 0

for i in range(itr):
    ep = np.random.uniform()
    
    # Exploitation, run the best arm
    if ep>e:
        # Checking for the best arm
        arm = np.argmax(egoi_mean)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_egoi += rev
        basket_egoi[i] = rev
        
        # Increase counter and recalculate mean
        egoi_counter[arm] += 1
        egoi_mean[arm] += (rev - egoi_mean[arm])/(egoi_counter[arm])
                    
    # Exploration, randomly select an arm
    else:
        arm = np.random.randint(0,3)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_egoi += rev
        basket_egoi[i] = rev
        
        # Increase counter and recalculate mean
        egoi_counter[arm] += 1
        egoi_mean[arm] += (rev - egoi_mean[arm])/(egoi_counter[arm])
        
    
# =============================================================================
# Epsilon-greedy algorithm with decay
# =============================================================================
# Initialise counters
# 0:lower, 1:middle, 2:higher
egd_counter = np.zeros(k)

# Initialise mean to be 100 for each arm
egd_mean = [100.0]*k
egd_mean = np.array(egd_mean)

# Set epsilon using this formula
# e = 1/(1+n/k) where k is the number of arms and n is the number of iterations already passed

# Overall revenue for this algorithm
revenue_egd = 0
basket_egd = np.zeros(itr)

for i in range(itr):
    ep = np.random.uniform()
    
    # Exploitation, run the best arm
    if ep > 1/(1+i/k):
        # Checking for the best arm
        arm = np.argmax(egd_mean)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_egd += rev
        egd_counter[arm] += 1
        basket_egd[i] = rev
        
        # Recalculate mean
        egd_mean[arm] += (rev - egd_mean[arm])/(egd_counter[arm] + 1) # Since we have initialisation =/= 0
 
    # Exploration, randomly select an arm
    else:
        arm = np.random.randint(0,k)
        
        # Calculate revenue and add to basket
        rev = np.sum(truedemand[arm] * prices[arm])
        revenue_egd += rev
        egd_counter[arm] += 1
        basket_egd[i] = rev
        
        # Recalculate mean
        egd_mean[arm] += (rev - egd_mean[arm])/(egd_counter[arm] + 1) # Since we have initialisation =/= 0
        
# =============================================================================
# UCB1-Tuned
# =============================================================================
# Initialise counters
# 0:lower, 1:middle, 2:higher
ucbt_counter = np.ones(k)

# Initialise basket
basket_ucbt = np.zeros(itr)

# UCB scores
ucbt_scores = np.zeros(k)

# sum of squares for ucb1-tuned
ucbt_ss = np.zeros(k)

# Initialise mean, pull each arm once
ucbt_mean = np.zeros(k)
for arm in range(k):
    rev = np.sum(truedemand[arm] * prices[arm])
    ucbt_mean[arm] = rev
    basket_ucbt[arm] = rev
    ucbt_ss[arm] += rev**2
    
# Update ucb scores
for arm in range(k):
    ucbt_scores[arm] = ucbt_mean[arm] + sqrt((log(k)/ucbt_counter[arm]) * min(1/4, (ucbt_ss[arm] + 2*(log(k)/ucbt_counter[arm]))) )

# Overall revenue
revenue_ucbt = np.sum(ucbt_mean)

for i in range(k, itr):
    # Find the arm with highest ucb score
    arm = np.argmax(ucbt_scores)
    
    # Calculate revenue and add to basket
    rev = np.sum(truedemand[arm] * prices[arm])
    revenue_ucbt += rev
    basket_ucbt[i] = rev
    
    # Increase counter and recalculate mean, sum of squares
    ucbt_counter[arm] += 1
    ucbt_mean[arm] += (rev - ucbt_mean[arm])/(ucbt_counter[arm])
    ucbt_ss[arm] += rev**2
    
    # Recalculate ucb score
    ucbt_scores[arm] = ucbt_mean[arm] + sqrt((log(i+1)/ucbt_counter[arm]) * min(1/4, (ucbt_ss[arm] + 2*(log(i+1)/ucbt_counter[arm]))) ) # i+1 for number of iterations
           
# Graphical comparison of results
plt.plot(np.cumsum(basket_cTS),'y')
plt.plot(np.cumsum(basket_eg),'b')
plt.plot(np.cumsum(basket_egd),'c')
plt.plot(np.cumsum(basket_egoi),'m')
plt.plot(np.cumsum(basket_ucb),'r')
plt.plot(np.cumsum(basket_ucbt),'k')
plt.ylabel('Cumulated revenue',fontsize=15)
plt.xlabel('Time period',fontsize=15)
plt.legend(['classical Thompson sampling','epsilon-greedy','epsilon-greedy w/ decay','epsilon-greedy w/ optimistic initialisation','ucb1','ucb1-tuned'],fontsize=20)
plt.show()
    

haha
# =============================================================================
# Thomson sampling (Dynamic pricing approach)
# =============================================================================
# Initialising prior distribution

# PED depends on the sequence of price vector used. Assume low-normal-high is the sequence. 
# normal - low
temp1 = (((xprice-xpricelower)/xpricelower)/((productprice-productpricelower)/productpricelower)).iloc[:,1]
# high - normal
temp2 = (((xpricehigher-xprice)/xprice)/((productpricehigher-productprice)/productprice)).iloc[:,1]

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
    








