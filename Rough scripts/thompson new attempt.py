# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 04:39:06 2019

@author: Samuel

In this script, I try to use the data generating mechanism as a start point,
instead of using it to generate prior data.

run 10 trials and compare with mas-rev-passive?
implement on original problem?

why mosek and cplex both cmi when f < 0 is not implemented? dont care. 
"""

numvars = 100

# Initialisation
import pandas as pd
import numpy as np
import sys, mosek
from scipy.optimize import minimize
from scipy.optimize import Bounds
import cplex
import matplotlib.pyplot as plt

iterations = 1
revenue_iterations = []
for ite in range(iterations):
    import pandas as pd
    import numpy as np
    import sys, mosek
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    import cplex

    # True parameter
    gammastar = np.random.uniform(-3,-1,numvars)
    gammastar = np.reshape(gammastar, (numvars,1))
    
    beta = 0.5 # original 0.5
    price0 = [12]*numvars
    price0 = np.array(price0)
    price0 = np.reshape(price0, (numvars,1))
    
    # Initial demand
    f1 = np.random.uniform(0.5,5,numvars)
    
    c0 = 0.05 # original 0.005
    
    noisemean = np.zeros(numvars)
    noisecov = np.identity(numvars)
    
    # For d(i,0)
    data = (f1 - c0 - np.random.multivariate_normal(noisemean, noisecov, 1))/beta 
    for i in range(numvars):
        if data[0,i] < 0:
            data[0,i] = 0
    data = np.reshape(data, (1,numvars))
     
    prevprice = price0
    
    # Triple data storage
    tripledata = []
    revenue_basket = []
    
    datapts = 100
    
    # Prior estimate of gamma, use as mean of distribution
    gammaprior = np.random.uniform(-8,-1,numvars)
    # constant c
    c = 0.1 * np.mean(gammaprior)
    elastmean = np.reshape(np.array(gammaprior),(numvars,1))
    elastcov = c*np.identity(numvars)
    
    historical_rev = np.zeros(2)
    historical_rev[0] = np.matmul(data, price0)
    sighat = np.std(historical_rev)
    
    # Data generating
    for i in range(1,datapts):
        # Demand forecast
        if i == 1:
            f = f1
        else:
            noise = np.random.multivariate_normal(noisemean, noisecov, 1)
            noise = np.reshape(noise, (numvars,1))
            f = c0 + noise
    
            # To calculate f
            for j in range(1, i+1):
                f += (beta**j)*np.reshape(data[i-j], (numvars,1))
                
    #        # To remove negative f
    #        for j in range(numvars):
    #            if f[j][0] < 0:
    #                f[j][0] = 0
    
        f = np.reshape(f, (numvars,1))
    
        # Random sample for elasticities
        elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)    
        # Ensures that all components are negative 
        while (elast<0).all() == False: 
            print("elast is positive")
            print(i)
            elast = np.random.multivariate_normal(elastmean.flatten(), elastcov,1)
        elast = np.reshape(elast, (numvars,1))
        
        # Generate price
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
    #        
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
    #                # Optimize
    #                task.optimize()
    #                
    #                # Print a summary containing information
    #                # about the solution for debugging purposes
    ##                task.solutionsummary(mosek.streamtype.msg)
    ##                task.analyzeproblem(mosek.streamtype.msg)
    #    
    #                # Output a solution
    #                xx = [0.] * numvar
    #                task.getxx(mosek.soltype.itr,
    #                           xx)
    #                
    ##                print("===== Check here =====")
    ##                # To return quadratic coefficients
    ##                qtemp = np.empty([numvar,numvar])
    ##                for row in range(numvar):
    ##                    for col in range(numvar):
    ##                        qtemp[row][col] = task.getqobjij(row,col)
    ##                        
    ##                print("no. of constraints =",task.getnumcon())
    ##                print("no. of nonzero elements in quadratic objective terms =",task.getnumqobjnz())
    ##                print("no. of cones =",task.getnumcone())
    ##                print("no. of variables =",task.getnumvar())
    ##                print("Objective sense =",task.getobjsense())
    ##                print("Problem type =",task.getprobtype())
    ##                print("Problem status =",task.getprosta(mosek.soltype.itr)) # feasible
    ##                print("var bound =",task.getvarbound(0))
    ##                print("===== End of check =====")
    ##                
    ##                # Get variable bounds
    ##                varbound = []
    ##                for b in range(numvar):
    ##                    varbound.append(task.getvarbound(b))
    ##                    
    ##                # Linear constraint
    ###                for index in range(numvar):
    ###                    print("no. of nonzero elements in {}-th column of A = {}".format(index,task.getacolnumnz(index)))
    ##                      
    ##                # To return linear coefficients
    ##                lineartemp = np.empty([numvar,1])
    ##                for index in range(numvar):
    ##                    lineartemp[index] = task.getcj(index)
    #
    ##                return (xx, qtemp, lineartemp, varbound)
    #                return xx
    #    
    #    # call the main function
    ##    result_mosek = main()
    ##    linear_coeff = result_mosek[2]
    ##    linear_check = f - (f * gammastar)
    ##    quad_coeff = result_mosek[1]
    ##    quad_check = 2 * f * gammastar / prevprice
    ##    var_bounds = result_mosek[3]
    ##    newprice = result_mosek[0]
    ##    newprice = np.array(newprice)
    ##    newprice = np.reshape(newprice, (numvars,1))
    #    
    #    newprice = main()
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
        
    
        # Observed demand
        observedx = f * (newprice / prevprice)**gammastar + np.reshape(np.random.multivariate_normal(noisemean, noisecov, 1), (numvars,1))
        for k in range(len(observedx)):
            if observedx[k,0] < 0:
                observedx[k,0] = 0
        
        # Append new data    
        data = np.append(data, np.transpose(observedx), axis=0)
    
        # Add data as triplet into tripledata
        tripledata.append([f, prevprice, observedx, elastmean])
        revenue_basket.append(np.sum(np.multiply(observedx, newprice)))

        if i == 1:
            historical_rev[1] = np.sum(np.multiply(observedx, newprice))
            sighat = np.std(historical_rev)
        
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
    
    
    
    
    
    priceshistory = []
    for i in range(len(tripledata)):
        priceshistory.append(tripledata[i][1])
    
    fhistory = []
    for i in range(len(tripledata)):
        fhistory.append(tripledata[i][0])
        
    revenue_iterations.append(revenue_basket)


plt.plot(revenue_basket)


