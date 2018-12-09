# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

#x=pd.read_excel("C:/Users/Samuel/Desktop/uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx")
x=pd.read_excel("C:/Uninotes/FYP/selected-sales data_children%27s book_every 99 cut 50.xlsx")

# removed brand_id, agio_cut_price and free_cut_price
dataoriginalprice=x.iloc[:,[1,17]]
datafullcut=x.iloc[:,[1,19]]

# merge based on order id for cut_price
datafullcut=datafullcut.groupby(['order_id']).sum().reset_index()

# merge based on order id for price per order
dataoriginalprice=dataoriginalprice.groupby(['order_id']).sum().reset_index()

# merge both together, sort by order amount
datamerge=pd.merge(dataoriginalprice,datafullcut)
datamerge=datamerge.sort_values(by='goods_money')

# remove outlier index 1139 where cut_goods_money = 0.35
datamerge.drop(1139,inplace=True)

###############################################################################################################################################
# checking for full-cut pricing

#import matplotlib.pyplot as plt
## %matplotlib qt
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

#plt.plot(mylist2,'r', mylist5,'b', mylist,'g', mylist4,'m', mylist3, 'y')
#plt.ylabel('Mean discount',fontsize=15)
#plt.xlabel('Groups',fontsize=15)
#plt.legend(['X = 80','X = 90','X = 99','X = 110', 'X = 120'],fontsize=20)

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
productpricelower=productprice-5
productpricehigher=productprice+5

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
###########################################################################################

np.random.seed(10)
v=np.random.randint(1,10, size=66)

x1=np.empty([66,1])
for i in range(66):
    x1[i]=math.exp(v[i]-(productprice.iloc[:,1])[i])
x1=x1/(sum(x1)+math.exp(5))

#teteijiosjriojiojio
#fjpajpdjsapjadps










