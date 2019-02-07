# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#x=pd.read_excel("C:/Users/Samuel/Desktop/uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx")
<<<<<<< HEAD
x = pd.read_excel("C:/Uninotes/FYP/data/selected-sales data_children%27s book_every 99 cut 50.xlsx")
=======
x=pd.read_excel("C:/Users/Samuel/Desktop/uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx")
>>>>>>> bd0db23227847dc3c4df1c4457642ce3b1c94a59

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

# x.nunique() tells us there are 66 unique goods   
# obtain price for these 66 goods     
productprice.drop_duplicates(subset="goods_id", keep='first', inplace=True)
productprice.reset_index(level=None, drop=True, inplace=True)

# create 2 other price vectors
productpricelower = productprice-5
productpricehigher = productprice+5

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
#    test1 = np.reshape((np.transpose(product[i,:])-productmean), (66,1))
#    test2 = np.reshape((np.transpose(product[i,:])-productmean), (1,66))
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
v = np.empty([66,1])
#np.random.seed(10)
for i in range(66):
    v[i] = np.random.randint(productprice.iloc[i,1]-0.05*productprice.iloc[i,1], productprice.iloc[i,1]+0.05*productprice.iloc[i,1])
#np.random.seed(10)
v0=np.random.randint(0, 5)
"""np.random.randint got size arg. can we use array operations instead of looping?"""
# Theoretical X, where X is the demand
xprice = np.empty([66,1])
xpricehigher = np.empty([66,1])
xpricelower = np.empty([66,1])

for i in range(66):
    xprice[i] = math.exp(v[i]-(productprice.iloc[:,1])[i])
# x0 is the choice where they dont buy anything
x0 = math.exp(v0)/(sum(xprice)+math.exp(v0))
xprice = xprice/(np.sum(xprice)+math.exp(v0))

#jaja = np.empty([66,1])
#jaja = math.exp(v-productprice.iloc[:,1])
#jaja = jaja/(np.sum(jaja)+math.exp(v0))



for i in range(66):
    xpricehigher[i] = math.exp(v[i]-(productpricehigher.iloc[:,1])[i])
x0higher = math.exp(v0)/(sum(xpricehigher)+math.exp(v0))
xpricehigher = xpricehigher/(np.sum(xpricehigher)+math.exp(v0))

for i in range(66):
    xpricelower[i] = math.exp(v[i]-(productpricelower.iloc[:,1])[i])
x0lower = math.exp(v0)/(sum(xpricelower)+math.exp(v0))
xpricelower = xpricelower/(np.sum(xpricelower)+math.exp(v0))

# =============================================================================
# To generate additional data by creating epsilons for each price vector
# =============================================================================
numep = 25
# First price vector
ep1 = np.empty([66,numep])
#np.random.seed(10)
for i in range(66):
    ep1[i] = np.random.uniform(low=0.0, high=xprice[i]/5, size=numep)

# create data via +epsilon and -epsilon so that the mean is still the theoretical mean
temp1 = np.empty([66,numep])
temp2 = np.empty([66,numep])
for i in range(66):
    temp1[i] = xprice[i]+ep1[i]
    temp2[i] = xprice[i]-ep1[i]
datax = np.append(temp1, temp2, axis=1)

# Second price vector
ep2 = np.empty([66,numep])
#np.random.seed(10)
for i in range(66):
    ep2[i] = np.random.uniform(low=0.0, high=xpricelower[i]/5, size=numep)

temp1 = np.empty([66,numep])
temp2 = np.empty([66,numep])
for i in range(66):
    temp1[i] = xpricelower[i]+ep2[i]
    temp2[i] = xpricelower[i]-ep2[i]
dataxlower = np.append(temp1, temp2, axis=1)

# Third price vector
ep3 = np.empty([66,numep])
#np.random.seed(10)
for i in range(66):
    ep3[i] = np.random.uniform(low=0.0, high=xpricehigher[i]/5, size=numep)

temp1 = np.empty([66,numep])
temp2 = np.empty([66,numep])
for i in range(66):
    temp1[i] = xpricehigher[i]+ep3[i]
    temp2[i] = xpricehigher[i]-ep3[i]
dataxhigher=np.append(temp1, temp2, axis=1)

# Appending all data points together for second approach
dataall = np.append(dataxlower, datax, axis=1)
dataall = np.append(dataall, dataxhigher, axis=1)

# Obtain revenue of each day, find SD to estimate sigma in second approach
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

# =============================================================================
# Thompson sampling done here (classical approach)
# =============================================================================

# Assume each price vector has normal distribution. Use MLE to estimate parameters from data created.
# First price vector
productmean = np.mean(datax, axis=1)
productcov = 0
for i in range(datax.shape[1]):
    test1 = np.reshape((datax[:,i]-productmean), (66,1))
    test2 = np.reshape((datax[:,i]-productmean), (1,66))
    productcov += np.matmul(test1, test2)
productcov = productcov/(i+1)
# divide by i+1 since MLE estimate of cov matrix is divide by N not N-1

# Second price vector
productmeanlower = np.mean(dataxlower, axis=1)
productcovlower = 0
for i in range(dataxlower.shape[1]):
    test1 = np.reshape((dataxlower[:,i]-productmeanlower), (66,1))
    test2 = np.reshape((dataxlower[:,i]-productmeanlower), (1,66))
    productcovlower += np.matmul(test1, test2)
productcovlower = productcovlower/(i+1)

# Third price vector
productmeanhigher = np.mean(dataxhigher, axis=1)
productcovhigher = 0
for i in range(dataxhigher.shape[1]):
    test1 = np.reshape((dataxhigher[:,i]-productmeanhigher), (66,1))
    test2 = np.reshape((dataxhigher[:,i]-productmeanhigher), (1,66))
    productcovhigher += np.matmul(test1, test2)
productcovhigher = productcovhigher/(i+1)

# Initialise counter for the number of times each arm is selected
counter = 0
counterlower = 0
counterhigher = 0
realrevenue_one = 0
realrevenuearr = np.zeros(1000)

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
#np.random.seed(10)
for i in range(1000):
    # Randomly sample from each distribution. This is our SAMPLED demand.
    rdm = np.random.multivariate_normal(productmean, productcov,1).T
    rdmlower = np.random.multivariate_normal(productmeanlower, productcovlower,1).T
    rdmhigher = np.random.multivariate_normal(productmeanhigher, productcovhigher,1).T
    
    # Calculate revenue based on the SAMPLED demand
    rev = np.multiply(rdm, productprice.iloc[:,1].values.reshape((66,1)))
    revlower = np.multiply(rdmlower, productpricelower.iloc[:,1].values.reshape((66,1)))
    revhigher = np.multiply(rdmhigher, productpricehigher.iloc[:,1].values.reshape((66,1)))
    
    # Choose the arm with the highest revenue based on SAMPLED demand.
    if np.sum(rev)>np.sum(revhigher) and np.sum(rev)>np.sum(revlower):
        counter += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
        realrevenue_one += np.sum(np.multiply(xprice, productprice.iloc[:,1].values.reshape((66,1))))
        realrevenuearr[i] += np.sum(np.multiply(xprice, productprice.iloc[:,1].values.reshape((66,1))))
        
        # Adding observed/theoretical X to list of observations
        datax = np.append(datax, xprice, axis=1)
        
        # Recalculate parameters using MLE
        productmean = np.mean(datax, axis=1)
        productcov = 0
        for i in range(datax.shape[1]):
            test1 = np.reshape((datax[:,i]-productmean), (66,1))
            test2 = np.reshape((datax[:,i]-productmean), (1,66))
            productcov += np.matmul(test1, test2)
        productcov = productcov/(i+1)

    elif np.sum(revlower)>np.sum(rev) and np.sum(revlower)>np.sum(revhigher):
        counterlower += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
        realrevenue_one += np.sum(np.multiply(xpricelower, productpricelower.iloc[:,1].values.reshape((66,1))))
        realrevenuearr[i] += np.sum(np.multiply(xpricelower, productpricelower.iloc[:,1].values.reshape((66,1))))
        
        # Adding observed/theoretical X to list of observations
        dataxlower = np.append(dataxlower, xpricelower, axis=1)
        
        # Recalculate parameters using MLE
        productmeanlower = np.mean(dataxlower, axis=1)
        productcovlower = 0
        for i in range(dataxlower.shape[1]):
            test1 = np.reshape((dataxlower[:,i]-productmeanlower), (66,1))
            test2 = np.reshape((dataxlower[:,i]-productmeanlower), (1,66))
            productcovlower += np.matmul(test1, test2)
        productcovlower = productcovlower/(i+1)

    else:
        counterhigher += 1
        
        # Pull the arm, calculate and accumulate OBSERVED revenue
        realrevenue_one += np.sum(np.multiply(xpricehigher, productpricehigher.iloc[:,1].values.reshape((66,1))))
        realrevenuearr[i] += np.sum(np.multiply(xpricehigher, productpricehigher.iloc[:,1].values.reshape((66,1))))
        
        # Adding observed/theoretical X to list of observations
        dataxhigher = np.append(dataxhigher, xpricehigher, axis=1)
        
        # Recalculate parameters using MLE
        productmeanhigher = np.mean(dataxhigher, axis=1)
        productcovhigher = 0
        for i in range(dataxhigher.shape[1]):
            test1 = np.reshape((dataxhigher[:,i]-productmeanhigher), (66,1))
            test2 = np.reshape((dataxhigher[:,i]-productmeanhigher), (1,66))
            productcovhigher += np.matmul(test1, test2)
        productcovhigher = productcovhigher/(i+1)


# =============================================================================
# Validate if the chosen arm is correct (use theoretical X for each arm)
# =============================================================================
revenue = 0
revenuearr = np.zeros(1000)
for i in range(1000):
    rev = np.multiply(xprice, productprice.iloc[:,1].values.reshape((66,1)))
    revenue += np.sum(rev)
    revenuearr[i] += np.sum(rev)

revenuelower = 0
revenuelowerarr = np.zeros(1000)
for i in range(1000):
    rev = np.multiply(xpricelower, productpricelower.iloc[:,1].values.reshape((66,1)))
    revenuelower += np.sum(rev)
    revenuelowerarr[i] += np.sum(rev)

revenuehigher = 0
revenuehigherarr = np.zeros(1000)
for i in range(1000):
    rev = np.multiply(xpricehigher, productpricehigher.iloc[:,1].values.reshape((66,1)))
    revenuehigher += np.sum(rev)
    revenuehigherarr[i] += np.sum(rev)

# Graphical comparison of results
#plt.plot(np.cumsum(realrevenuearr),'r')
#plt.plot(np.cumsum(revenuelowerarr),'b')
#plt.plot(np.cumsum(revenuearr),'y')
#plt.plot(np.cumsum(revenuehigherarr),'m')
#plt.ylabel('Cumulated revenue',fontsize=15)
#plt.xlabel('Time period',fontsize=15)
#plt.legend(['Real revenue','Lower arm','Middle arm','Higher arm'],fontsize=20)
#plt.show()




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
elastmean = np.reshape(np.array(elastmean),(66,1))
elastcov = c*np.identity(66)

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
prevprice = np.reshape(prevprice, (66,1))

realrevenue_two = 0

#f = results.forecast(datadiff.values[-lag_order:], 1) # with diff and log
#f = np.reshape(f, (66,1))
#f = np.e**(f + np.log(np.reshape(np.array(dataall.iloc[149,:]), (66,1))))

#f = results.forecast(datadiff.values[-lag_order:], 1) # with abs(diff) THEN log
#f = np.reshape(f, (66,1))
#f = np.e**(f) + np.reshape(np.array(dataall.iloc[149,:]), (66,1))

# testing for cointegration
#from statsmodels.tsa.vector_ar.vecm import coint_johansen
#haha = dataall.iloc[:,range(60,66)]
#coint_johansen(haha,-1,1).eig

import cplex
from scipy.optimize import minimize
from scipy.optimize import Bounds
# Set bounds to be lower prices and higher prices
bounds = Bounds(np.array(productpricelower.iloc[:,1]), np.array(productpricehigher.iloc[:,1]))
from scipy.optimize import differential_evolution
import sys, os, mosek






for j in range(100):
    print(j)
    # Random sample for elasticities
    elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
    # Ensures that all components are negative 
    while (elast<0).all() == False: 
        print("elast is positive")
        print(j)
        elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
    elast = np.reshape(elast, (66,1))
    print("random sample ok")
    
    # Obtain demand forecast f and reshape
    """without diff and log"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (66,1))
    """diff()"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (66,1))
#    f = f + np.reshape(np.array(dataall.iloc[-lag_order,:]), (66,1)) 
    """diff(log)"""
#    f = results.forecast(datadiff.values[-lag_order:], 1)
#    f = np.reshape(f, (66,1))
#    f = np.e**(f + np.log(np.reshape(np.array(dataall.iloc[-lag_order,:]), (66,1))))  
    """log(abs(diff))"""
    f = results.forecast(datadiff.values[-lag_order:], 1)
    f = np.reshape(f, (66,1))
    f = np.e**(f) + np.reshape(np.array(dataall.iloc[-lag_order,:]), (66,1))
    
    # Checking for valid f
    if (f>=0).all() == False:
        print("f is negative")
        break
    print("demand forecast ok")

    """using mosek"""
    # Since the actual value of Infinity is ignored, we define it solely for symbolic purposes:
    inf = 0.0

    # Define a stream printer to grab output from MOSEK
    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()
    
    def main():
    # Open MOSEK and create an environment and task
    # Make a MOSEK environment
        with mosek.Env() as env:
            # Attach a printer to the environment
            env.set_Stream(mosek.streamtype.log, streamprinter)
            # Create a task
            with env.Task() as task:
                task.set_Stream(mosek.streamtype.log, streamprinter)
                
                # Bound keys for variables
                numvar = 66
                bkx = [mosek.boundkey.ra] * numvar
                
                # Bound values for variables
                temppricelow = prevprice*0.9
                temppricehigh = prevprice*1.1
                blx = []
                bux = []
                for i in range(66):
                    blx.append(temppricelow[i][0])
                    bux.append(temppricehigh[i][0])
                
                # Objective linear coefficients
                temp = f - (f * elast)
                c = []
                for i in range(numvar):
                    c.append(temp[i][0])
#                print(c)
                print()
                # Append 'numvar' variables.
                # The variables will initially be fixed at zero (x=0).
                task.appendvars(numvar)
    
                for j in range(numvar):
                    # Set the linear term c_j in the objective.
                    task.putcj(j, c[j])
                    
                    # Set the bounds on variable j
                    # blx[j] <= x_j <= bux[j] 
                    task.putbound(mosek.accmode.var, j, bkx[j], blx[j], bux[j]) 
                        
                # Set up and input quadratic objective
                qsubi = []
                for i in range(numvar):
                    qsubi.append(i)
                qsubj = qsubi
                temp = f * elast / prevprice
                qval = []
                for i in range(numvar):
                    qval.append(temp[i][0])
#                print(qval)
    
                task.putqobj(qsubi, qsubj, qval)
    
                # Input the objective sense (minimize/maximize)
                task.putobjsense(mosek.objsense.maximize)
    
                # Optimize
                task.optimize()
                
                # Print a summary containing information
                # about the solution for debugging purposes
#                task.solutionsummary(mosek.streamtype.msg)
    
                # Output a solution
                xx = [0.] * numvar
                task.getxx(mosek.soltype.itr,
                           xx)
    
                return xx
     # call the main function
    newprice = main()
    newprice = np.array(newprice)
    
<<<<<<< HEAD
    """using scipy.optimize"""    
=======
    """using scipy.optimize"""
>>>>>>> bd0db23227847dc3c4df1c4457642ce3b1c94a59
#    # Objective function, multiply by -1 since we want to maximize
#    def eqn7(p):
#        return -1.0*np.sum(p*p*f*elast/prevprice - p*f*elast + p*f)
#    
#    # Initial guess is previous price
#    opresult = minimize(eqn7, np.reshape(np.array(productpricehigher.iloc[:,1]), (66,1)), bounds=bounds)
#    newprice = opresult.x
    
    """using cplex"""
#    # create an instance
#    problem = cplex.Cplex()
#    
#    # set the function to maximise instead of minimise
#    problem.objective.set_sense(problem.objective.sense.maximize)
#    
#    # Adds variables
#    indices = problem.variables.add(names = [str(i) for i in range(66)])
#    
#    # Changes the linear part of the objective function.
#    for i in range(66):
#        problem.objective.set_linear(i, float(f[i]-f[i]*elast[i])) # form is objective.set_linear(var, value)
#        
#    # Sets the quadratic part of the objective function.
#    quad = (f*elast/prevprice)
#    problem.objective.set_quadratic([float(i) for i in quad])
#    
#    # Sets the lower bound for a variable or set of variables
#    for i in range(66):
#        problem.variables.set_lower_bounds(i, prevprice[i][0]*0.9)
#    
#    # Sets the upper bound for a variable or set of variables
#    for i in range(66):
#        problem.variables.set_upper_bounds(i, prevprice[i][0]*1.1)
#    
#    problem.solve()
#    newprice=problem.solution.get_values()
#    newprice = np.array(newprice)
    print("optimization ok")
    
    
    
    
    # Apply newprice to obtain observed demand
    observedx = np.empty([66,1])
    for i in range(66):
        observedx[i] = math.exp(v[i]-newprice[i])
    observedx = observedx/(np.sum(observedx)+math.exp(v0))
    print("observed demand ok")
    
    # Accumulate revenue
    realrevenue_two += np.sum(np.multiply(observedx, np.reshape(newprice,(66,1))))
    
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
    thet = np.multiply(np.reshape(newprice**2,(66,1)),f)
    thet = np.divide(thet, prevprice)
    thet = thet - np.multiply(np.reshape(newprice,(66,1)),f)
    minv = (thet*np.transpose(thet))/sighat**2 + 1e-5*np.identity(66) # fix lambda = 1e-5
    print("minv ok")
    
    # For M inverse beta matrix
    rbar = np.sum(np.multiply(np.reshape(newprice,(66,1)),f))
    rt = float(np.sum(np.multiply(observedx, np.reshape(newprice,(66,1)))))
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
    prevprice = np.reshape(newprice, (66,1))
    








