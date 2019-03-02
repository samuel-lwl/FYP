# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:20:03 2019

@author: Samuel
"""
numvars = 10

# Initialisation
import pandas as pd
import numpy as np
np.random.seed(10)
# True parameter
gammastar = np.random.uniform(-3,-1,numvars)
gammastar = np.reshape(gammastar, (numvars,1))

beta = 0.0005
price0 = [12]*numvars
price0 = np.array(price0)
price0 = np.reshape(price0, (numvars,1))

np.random.seed(10)
# Initial demand
f1 = np.random.uniform(0.5,5,numvars)

c0 = 0.005
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

noisemean = np.zeros(numvars)
noisecov = np.identity(numvars)

# For d(i,0)
np.random.seed(10)
data = (f1 - c0 - np.random.multivariate_normal(noisemean, noisecov, 1))/beta 
for i in range(numvars):
    if data[0][i] < 0:
        data[0][i] = 0
data = np.reshape(data, (1,numvars))

prevprice = price0

import sys, mosek
from scipy.optimize import minimize
from scipy.optimize import Bounds
import cplex

# Triple data storage
tripledata = []

datapts = 20

# Data generating
for i in range(1,datapts):
    # Demand forecast
    if i == 1:
        f = f1
    else:
#        noise = np.random.multivariate_normal(noisemean, noisecov, 1) + np.reshape(noise, (numvars,1))
        f = c0 
        for j in range(1,i+1):
            f += (beta**j)*np.reshape(data[i-j], (numvars,1))
    f = np.reshape(f, (numvars,1))
    
    
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
                numvar = 10
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
                temp = f - (f * gammastar)
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
                temp = 2 * f * gammastar / prevprice # Must remember to *2, see mosek documentation
                qval = []
                for i in range(numvar):
                    qval.append(temp[i][0])
#                print(qval)
    
                task.putqobj(qsubi, qsubj, qval)
    
                # Input the objective sense (minimize/maximize)
                task.putobjsense(mosek.objsense.maximize)
                task.analyzeproblem(mosek.streamtype.log)
                # Optimize
                task.optimize()
                
                # Print a summary containing information
                # about the solution for debugging purposes
#                task.solutionsummary(mosek.streamtype.msg)
#                task.analyzeproblem(mosek.streamtype.msg)
    
                # Output a solution
                xx = [0.] * numvar
                task.getxx(mosek.soltype.itr,
                           xx)
                
                print("===== Check here =====")
                # To return quadratic coefficients
                qtemp = np.empty([numvar,numvar])
                for row in range(numvar):
                    for col in range(numvar):
                        qtemp[row][col] = task.getqobjij(row,col)
                        
                print("no. of constraints =",task.getnumcon())
                print("no. of nonzero elements in quadratic objective terms =",task.getnumqobjnz())
                print("no. of cones =",task.getnumcone())
                print("no. of variables =",task.getnumvar())
                print("Objective sense =",task.getobjsense())
                print("Problem type =",task.getprobtype())
                print("Problem status =",task.getprosta(mosek.soltype.itr)) # feasible
                print("var bound =",task.getvarbound(0))
                print("===== End of check =====")
                
                # Get variable bounds
                varbound = []
                for b in range(numvar):
                    varbound.append(task.getvarbound(b))
                    
                # Linear constraint
#                for index in range(numvar):
#                    print("no. of nonzero elements in {}-th column of A = {}".format(index,task.getacolnumnz(index)))
                      
                # To return linear coefficients
                lineartemp = np.empty([numvar,1])
                for index in range(numvar):
                    lineartemp[index] = task.getcj(index)

                return (xx, qtemp, lineartemp, varbound)
            
    # call the main function
    result_mosek = main()
    linear_coeff = result_mosek[2]
    linear_check = f - (f * gammastar)
    quad_coeff = result_mosek[1]
    quad_check = 2 * f * gammastar / prevprice
    var_bounds = result_mosek[3]
    
    
    newprice = result_mosek[0]
    newprice = np.array(newprice)
    newprice = np.reshape(newprice, (numvars,1))

    """using scipy.optimize"""
#    # Objective function, multiply by -1 since we want to maximize
#    def eqn7(p):
#        return -1.0*np.sum(p*p*f.flatten()*gammastar.flatten()/prevprice.flatten() - p*f.flatten()*gammastar.flatten() + p*f.flatten())
#    
#    # Initial guess is 1.05 * previous price
#    bounds = Bounds(prevprice.flatten()*0.9, prevprice.flatten()*1.1)
#    opresult = minimize(eqn7, prevprice.flatten()*1.05, bounds=bounds)
#    newprice = opresult.x
#    newprice = np.reshape(newprice, (numvars,1))   
    
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
#        problem.objective.set_linear(i, float(f[i]-f[i]*gammastar[i])) # form is objective.set_linear(var, value)
#        
#    # Sets the quadratic part of the objective function.
#    quad = (f*gammastar/prevprice) # need to *2, see optimisation_test.py
#    problem.objective.set_quadratic([2*float(i) for i in quad])
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
#    newprice = problem.solution.get_values()
#    newprice = np.array(newprice)
#    newprice = np.reshape(newprice, (numvars,1))
        
    # Observed demand
    observedx = f * (newprice / prevprice)**gammastar + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k][0] < 0:
            observedx[k][0] = 0
    data = np.append(data, np.transpose(observedx), axis=0)

#    
    # Add data as triplet into tripledata
    tripledata.append([f, prevprice, observedx])
    
    # Update prevprice to new price
    prevprice = newprice


priceshistory = []
for i in range(len(tripledata)):
    priceshistory.append(tripledata[i][1])

fhistory = []
for i in range(len(tripledata)):
    fhistory.append(tripledata[i][0])


#############################################################################



# Prior estimate of gamma, use as mean of distribution
np.random.seed(100)
gammaprior = np.random.uniform(-5,-1,numvars)
# constant c
c = 0.1 * np.mean(gammaprior)
elastmean = np.reshape(np.array(gammaprior),(numvars,1))
elastcov = c*np.identity(numvars)

sighat = np.std(data)

# Preparing data for var model, need to insert date-time 
dataall = data
dataall = pd.DataFrame(dataall)
dataall.insert(0, 'date_time', list(range(datapts)))
dataall.iloc[:,0] = pd.to_datetime(dataall.iloc[:,0]) # converting date_time column to datetime64[ns] data type
dataall.index = dataall.date_time # change index to datetime since index must be datetime to run time series models
dataall = dataall.drop(['date_time'], axis=1) # remove datetime from column
#dataall.dtypes # to check if date_time is datetime64[ns]




# Making data stationary
#datadiff = np.log(dataall).diff().dropna() # diff(log)
#datadiff = dataall.diff().dropna() # diff()
#datadiff = dataall # without diff and log
datadifforg = dataall.diff().dropna() # log(abs(diff))
datadiff = np.log(abs(datadifforg)) # log(abs(diff))

datadiff = datadiff.replace([np.inf, -np.inf], 0) # replace inf with 0

# Time series model for forecasting
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import AR

# Run a VAR model on DIFFERENCED data
model = VAR(datadiff) 
#model = AR(datadiff.iloc[:,0])

# Estimate the coefficients of the VAR model
# select_order should show the information criterion for each number of lag
#model.select_order() # >1 lag then matrix is not positive definite. w/o diff and log, cant even do lag 1.
results = model.fit()
#results = model.fit(maxlag=(len(datadiff)-1), ic='aic')

# lag_order is the best lag chosen by model.fit()
lag_order = results.k_ar

#oof=results.predict(start=len(datadiff), end=len(datadiff))
#oof=results.forecast(datadiff.values[-lag_order:], steps=1)

from cvxopt import matrix, solvers

revenue_basket = []

# Implementing TS
for j in range(40):
    print(j)
    # Random sample for elasticities
    np.random.seed(40)
    elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
    # Ensures that all components are negative 
    while (elast<0).all() == False: 
        print("elast is positive")
        print(j)
        np.random.seed(50)
        elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
    elast = np.reshape(elast, (numvars,1))
    print("random sample ok")
    
#    f = np.zeros((1,100))
#    for var in range(numvars):
#        if var == 0:
#            model = AR(datadiff.iloc[:,0])
#        else:
#            datadiff.iloc[:,0] = datadiff.iloc[:,var]
#            model = AR(datadiff.iloc[:,0])
#        results = model.fit(maxlag=(len(datadiff)-1), ic='aic')
#        temp = results.predict(start=len(datadiff), end=len(datadiff))
#        f[0][var] = temp[0]
#        
#    f = np.reshape(f, (numvars,1)) 
    
    # Demand forecast
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
#        print("f is negative")
#        break
        for i in range(len(f)):
            if f[i][0] < 0:
                f[i][0] = 0        
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
                numvar = numvars
                bkx = [mosek.boundkey.ra] * numvar
#                bkx = [mosek.boundkey.lo] * numvar
                
                # Bound values for variables
                temppricelow = prevprice*0.9
                temppricehigh = prevprice*1.1
                blx = []
#                bux = [inf] * numvar
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
    
    """using cvxopt"""
#    temp = -2 * f * elast / prevprice
#    P = np.diagflat(np.array([x[0] for x in temp]))
#    q = f - (f * elast)
#    temp = np.array([-1] * numvars)
#    g = np.diagflat(temp)
#    h = prevprice*0.9
#    sol = solvers.qp(P=matrix(P), q=matrix(q), G=matrix(g, size=(10,10), tc='i'), h=matrix(h), solver='mosek')
#    
    """using scipy.optimize"""
#    # Objective function, multiply by -1 since we want to maximize
#    def eqn7(p):
#        return -1.0*np.sum(p*p*f.flatten()*elast.flatten()/prevprice.flatten() - p*f.flatten()*elast.flatten() + p*f.flatten())
#    
#    # Initial guess is 1.05 * previous price
#    bounds = Bounds(prevprice.flatten()*0.9, prevprice.flatten()*1.1)
#    opresult = minimize(eqn7, prevprice.flatten()*1.05, bounds=bounds)
#    newprice = opresult.x
#    newprice = np.reshape(newprice, (numvars,1))   
    
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
#    quad = (f*elast/prevprice) # need to *2, see optimisation_test.py
#    problem.objective.set_quadratic([2*float(i) for i in quad])
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
#    newprice = problem.solution.get_values()
#    newprice = np.array(newprice)
#    newprice = np.reshape(newprice, (numvars,1))
    
    print("optimization ok")
    
    # Observed demand
    observedx = f * (newprice / prevprice)**gammastar + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
    for k in range(len(observedx)):
        if observedx[k][0] < 0:
            observedx[k][0] = 0

    print("observed demand ok")
        
    # Accumulate revenue
    revenue_basket.append(np.sum(np.multiply(observedx, newprice)))
    
    # Append new data    
    observedx = pd.DataFrame(observedx)
    dataall = dataall.append(observedx.transpose())

    dataall.insert(0, 'date_time', list(range(datapts+j+1)))
    dataall.iloc[:,0] = pd.to_datetime(dataall.iloc[:,0]) # converting date_time column to datetime data type
    dataall.index = dataall.date_time # change index to datetime
    dataall = dataall.drop(['date_time'], axis=1) # remove datetime from column
        
#    datadiff = np.log(dataall).diff().dropna() # diff(log)
#    datadiff = dataall.diff().dropna() # diff()
#    datadiff = dataall # without diff and log
#
    datadifforg = dataall.diff().dropna() # log(abs(diff))
    datadiff = np.log(abs(datadifforg)) # log(abs(diff))
    datadiff = datadiff.replace([np.inf, -np.inf], 0) # replace inf with 0
    print("add demand ok")
            
    # Re-estimate VAR model
    model = VAR(datadiff)
    results = model.fit(10)
    lag_order = results.k_ar
    print("re-estimate var model ok")
    
    
    
    
    
    # For M inverse matrix
    thet = (newprice * newprice * f / prevprice) - (newprice * f)
    minv = (thet * np.transpose(thet)) / (sighat**2) + 1e-5 * np.identity(numvars) # fix lambda = 1e-5
    print("minv ok")
    
    # For M inverse beta matrix
    rbar = np.sum(newprice * f)
    rt = (np.sum(newprice * observedx))[0]
    minvb = (rt - rbar)/(sighat**2) * thet
    print("minvb ok")

    # Update mean of elasticity
    pt1 = np.linalg.inv(np.linalg.inv(elastcov) + minv)
    pt2 = np.matmul(np.linalg.inv(elastcov), elastmean) + minvb
    elastmean = np.matmul(pt1, pt2)
    print("update mean ok")
    
    # Update cov of elasticity
    elastcov = pt1
    
    tripledata.append([f, prevprice, observedx])
    
    # Update prevprice to new price
    prevprice = newprice
        
        
        
basket_mosek = revenue_basket

priceshistory_mosek = []
for i in range(len(tripledata)):
    priceshistory_mosek.append(tripledata[i][1])

fhistory_mosek = []
for i in range(len(tripledata)):
    fhistory_mosek.append(tripledata[i][0])

dataall_mosek = dataall



#basket_scipy = revenue_basket
#
#priceshistory_scipy = []
#for i in range(len(tripledata)):
#    priceshistory_scipy.append(tripledata[i][1])
#
#fhistory_scipy= []
#for i in range(len(tripledata)):
#    fhistory_scipy.append(tripledata[i][0])
#
#dataall_scipy = dataall
    
    
    
    
#basket_cplex = revenue_basket
#
#priceshistory_cplex = []
#for i in range(len(tripledata)):
#    priceshistory_cplex.append(tripledata[i][1])
#
#fhistory_cplex = []
#for i in range(len(tripledata)):
#    fhistory_cplex.append(tripledata[i][0])
#
#dataall_cplex = dataall
    

        
        
        
        
        
        
        
        
        



