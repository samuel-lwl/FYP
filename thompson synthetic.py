# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:20:03 2019

@author: Samuel
"""
numvars = 100

# Initialisation
import numpy as np
np.random.seed(10)
gammastar = np.random.uniform(-3,-1,numvars)

beta = 0.5
price0 = [12]*numvars
price0 = np.array(price0)
price0 = np.reshape(price0, (numvars,1))

np.random.seed(10)
f1 = np.random.uniform(0.5,5,numvars)

c0 = 0.05
#np.random.seed(1)
#data = np.random.uniform(0.5,5,100)
## each row is 1 observation, each column is 1 product
#data = np.reshape(data, (1,100))
#np.random.seed(2)
#for i in range(99):
#    data = np.append(data,np.reshape(np.random.uniform(0.5,5,100),(1,100)), axis=0)

# =============================================================================
# MAX-REV-TS
# =============================================================================

# Prior estimate of gamma, use as mean of distribution
np.random.seed(100)
gammaprior = np.random.uniform(-5,-1,numvars)
# constant c
c = 0.1 * np.mean(gammaprior)
elastmean = np.reshape(np.array(gammaprior),(numvars,1))
elastcov = c*np.identity(numvars)

# For d(i,0)
data = (f1 - c0)/beta 
data = np.reshape(data, (1,numvars))
sighat = np.std(data)

prevprice = price0

noisemean = np.zeros(numvars)
noisecov = np.identity(numvars)

import sys, mosek
from scipy.optimize import minimize
from scipy.optimize import Bounds

revenue = 0

for i in range(1,10):
    # Demand forecast
    if i == 1:
        f = f1
    else:
        noise = np.random.multivariate_normal(noisemean, noisecov, 1)
        f = c0 + np.reshape(noise, (numvars,1))
        for j in range(1,i+1):
            f += (beta**j)*np.reshape(data[i-j], (numvars,1))
    f = np.reshape(f, (numvars,1))
    
    # Elasticity estimate
    elast = np.random.multivariate_normal(elastmean.flatten(), elastcov, 1)
    # Ensures that all components are negative 
    while (elast<0).all() == False: 
        print("elast is positive")
        print(i)
        elast = np.random.multivariate_normal(elastmean.flatten(), elastcov, 1)
    elast = np.reshape(elast, (numvars,1))
    
    # Generate price
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
                numvar = 100
                bkx = [mosek.boundkey.ra] * numvar
                
                # Bound values for variables
                temppricelow = prevprice*0.9
                temppricehigh = prevprice*1.1
                blx = []
                bux = []
                for i in range(numvar):
                    blx.append(temppricelow[i][0])
                    bux.append(temppricehigh[i][0])
                
                # Objective linear coefficients
                temp = f - (f * elast)
                c = []
                for i in range(numvar):
                    c.append(temp[i][0])
#                print(c)
                
                # Append 'numcon' empty constraints.
                # The constraints will initially have no bounds.
                task.appendcons(0)
            
                # Append 'numvar' variables.
                # The variables will initially be fixed at zero (x=0).
                task.appendvars(numvar)
    
                for j in range(numvar):
                    # Set the linear term c_j in the objective.
                    task.putcj(j, c[j])
                    
                    # Set the bounds on variable j
                    # blx[j] <= x_j <= bux[j] 
                    task.putvarbound(j, bkx[j], blx[j], bux[j]) 
                        
                # Set up and input quadratic objective
                qsubi = []
                for i in range(numvar):
                    qsubi.append(i)
                qsubj = qsubi
                temp = 2 * f * elast / prevprice # Must remember to *2, see mosek documentation
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
    newprice = np.reshape(newprice, (numvars,1))
    
    """using scipy.optimize"""
#    bounds = Bounds(prevprice*0.9, prevprice*1.1)
#    
#    # Objective function, multiply by -1 since we want to maximize
#    def eqn7(p):
#        return -1.0*np.sum(p*p*f*elast/prevprice - p*f*elast + p*f)
#    
#    # Initial guess is previous price
#    opresult = minimize(eqn7, prevprice*1.05, bounds=bounds)
#    newprice = opresult.x
#    newprice = np.reshape(newprice, (numvars,1))
    
    
    
    
    
    
    
    
    # Observed demand
    observedx = f * (newprice / prevprice)**elast
    for k in range(len(observedx)):
        if observedx[k][0] < 0:
            observedx[k][0] = 0
    data = np.append(data, np.transpose(observedx), axis=0)

    # Accumulate revenue
    revenue += np.sum(observedx * newprice)

    # For M inverse matrix
    thet = (newprice * newprice * f / prevprice) - (newprice * f)
    minv = (thet * np.transpose(thet)) / (sighat**2) + 1e-5 * np.identity(numvars) # fix lambda = 1e-5

    # For M inverse beta matrix
    rbar = np.sum(newprice * f)
    rt = np.sum(newprice * observedx)
    minvb = (rt - rbar)/(sighat**2) * thet

    # Update mean of elasticity
    pt1 = np.linalg.inv(np.linalg.inv(elastcov) + minv)
    pt2 = np.matmul(np.linalg.inv(elastcov), elastmean) + minvb
    elastmean = np.matmul(pt1, pt2)
    
    # Update cov of elasticity
    elastcov = pt1
    
    # Update prevprice to new price
    prevprice = newprice
























