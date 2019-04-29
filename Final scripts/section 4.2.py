import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from math import sqrt
from math import log

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

# =============================================================================
# Generate market share X for each price vector
# =============================================================================
# where X is generated using a constant elasticity model

# Number of arms
k = 3

# Use day01 as baseline demand
day01 = x.loc[x['order_date']==20170804,:]
day01 = day01.groupby('goods_id',as_index=False)[["goods_amount"]].sum()
f0 = np.array(day01.iloc[:,1])
f0 = np.reshape(f0, (numvars,1))

gammastar = np.random.uniform(-3,-1,numvars)
gammastar = np.reshape(gammastar, (numvars,1))

# True demand: Theoretical X, where X is the demand
truedemand = [np.empty([numvars,1])] * k

for arm in range(k):
    truedemand[arm] = f0 * (prices[arm]/prices[1])**gammastar

# =============================================================================
# To generate additional data by creating epsilons for each price vector
# =============================================================================
# Number of epsilons. Multiply by 2 to get number of data points per arm. 
numep = 25

# To hold data for all arms
data_all = []

for arm in range(k):
    ep = np.empty([numvars,numep])
    for i in range(numvars):
        # choose epsilon to be within 0 to 5% of true demand
        ep[i] = np.random.uniform(low=0.0, high=truedemand[arm][i]*0.05, size=numep)
    
    # create data via +epsilon and -epsilon so that the mean is still the theoretical mean
    temp1 = np.empty([numvars,numep])
    temp2 = np.empty([numvars,numep])
    for i in range(numvars):
        temp1[i] = truedemand[arm][i] + ep[i]
        temp2[i] = truedemand[arm][i] - ep[i]
    data_all.append(np.append(temp1, temp2, axis=1))

# Number of iterations
itr = 1000

# =============================================================================
# Thompson sampling done here (classical approach)
# =============================================================================
# Assume each price vector has normal distribution. Use MLE to estimate parameters from data created.
# Mean for each arm
cTS_mean = [0.0]*k
for arm in range(k):
    cTS_mean[arm] = np.mean(data_all[arm], axis=1)

# Covariance
cTS_cov = [0.0]*k
for arm in range(k):
    # Must reset cov = 0 for each arm other cumulative sum will affect later arms
    cov = 0
    for i in range(data_all[arm].shape[1]): # For each data point of each arm
        temp1 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (numvars,1))
        temp2 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (1,numvars))
        cov += np.matmul(temp1, temp2)
        
    # divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
    cTS_cov[arm] = cov/(numep*2)

# Initialise counter for the number of times each arm is selected
cTS_counter = np.zeros(k)

# Revenue and basket
revenue_cTS = 0
basket_cTS = np.zeros(itr)

for j in range(itr):
    # Randomly sample from each distribution. This is our SAMPLED demand.
    cTS_forecast_demand = [0.0]*k
    for arm in range(k):
        cTS_forecast_demand[arm] = np.random.multivariate_normal(cTS_mean[arm], cTS_cov[arm], 1).T
    
    # Calculate revenue based on the SAMPLED demand
    cTS_forecast_revenue = np.zeros(k)
    for arm in range(k):
        cTS_forecast_revenue[arm] = np.sum(np.multiply(cTS_forecast_demand[arm], prices[arm]))
       
    # Choose the arm with the highest revenue based on SAMPLED demand.
    arm = np.argmax(cTS_forecast_revenue)
    
    # Add to counter
    cTS_counter[arm] += 1
    
    # Pull the arm, calculate and accumulate OBSERVED revenue
    rev = np.sum(np.multiply(truedemand[arm], prices[arm]))
    revenue_cTS += rev
    basket_cTS[j] = rev
    
    # Adding observed/theoretical X to list of observations
    data_all[arm] = np.append(data_all[arm], truedemand[arm], axis=1)
    
    # Recalculate parameters using MLE
    # Mean for each arm
    cTS_mean[arm] = np.mean(data_all[arm], axis=1)
    
    # Covariance
    cov = 0
    for i in range(data_all[arm].shape[1]): # For each data point of each arm
        temp1 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (numvars,1))
        temp2 = np.reshape((data_all[arm][:,i] - cTS_mean[arm]), (1,numvars))
        cov += np.matmul(temp1, temp2)       
    # divide by number of data points since MLE estimate of cov matrix is divide by N not N-1
    cTS_cov[arm] = cov/(data_all[arm].shape[1])

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
#plt.plot(np.cumsum(basket_cTS),'ko-') 
#plt.plot(np.cumsum(basket_real[:,0]),'k--')
#plt.plot(np.cumsum(basket_real[:,1]),'k-.')
#plt.plot(np.cumsum(basket_real[:,2]),'k:')
#plt.ylabel('Cumulated revenue',fontsize=20)
#plt.xlabel('Time period',fontsize=20)
#plt.legend(['Real revenue','Lower arm','Middle arm','Higher arm'],fontsize=20)
#plt.title("Comparing results with all 3 arms", fontsize=20)
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
plt.plot(np.cumsum(basket_cTS),'ko-')
plt.plot(np.cumsum(basket_eg),'k--')
plt.plot(np.cumsum(basket_egd),'k-.')
plt.plot(np.cumsum(basket_egoi),'k:')
plt.plot(np.cumsum(basket_ucb),'k+-.')
plt.plot(np.cumsum(basket_ucbt),'kx:')
plt.ylabel('Cumulated revenue',fontsize=20)
plt.xlabel('Time period',fontsize=20)
plt.legend(['classical Thompson sampling','epsilon-greedy','epsilon-greedy w/ decay','epsilon-greedy w/ optimistic initialisation','ucb1','ucb1-tuned'],fontsize=20)
plt.title("Comparing TS with other algorithms",fontsize=20)
plt.show()


