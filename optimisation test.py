# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:39:30 2019

@author: Samuel
"""
import cplex
import numpy as np
import sys, mosek
from scipy.optimize import minimize
from scipy.optimize import Bounds

numvars = 2

linearpart = np.array([5,2],dtype='float')
quadpart = np.array([-5, -5], dtype='float')
initial = np.array([1,1], dtype='float')

list_cplex = []
list_scipy = []
list_mosek = []

"""using cplex"""
for i in range(50):
    # create an instance
    problem = cplex.Cplex()
    
    # set the function to maximise instead of minimise
    problem.objective.set_sense(problem.objective.sense.maximize)
    
    # Adds variables
    indices = problem.variables.add(names = [str(i) for i in range(numvars)])
    
    # Changes the linear part of the objective function.
    for i in range(numvars):
        problem.objective.set_linear(i, linearpart[i]) # form is objective.set_linear(var, value)
        
    # Sets the quadratic part of the objective function.
    problem.objective.set_quadratic(2*quadpart) # Must *2 
    
    # Sets the lower bound for a variable or set of variables
    for i in range(numvars):
#        problem.variables.set_lower_bounds(i, -cplex.infinity)
        problem.variables.set_lower_bounds(i, initial[i]*0.9)
    
    # Sets the upper bound for a variable or set of variables
    for i in range(numvars):
#        problem.variables.set_upper_bounds(i, cplex.infinity)
        problem.variables.set_upper_bounds(i, initial[i]*1.1)
    
    problem.solve()
    newprice = problem.solution.get_values()
    newprice = np.array(newprice)
    newprice_cplex = np.reshape(newprice, (numvars,1))
    list_cplex.append(newprice_cplex)

    initial = np.array(newprice_cplex.flatten(), dtype='float')

initial = np.array([1,1], dtype='float')

"""using scipy.optimize"""
for i in range(50):
    # Objective function, multiply by -1 since we want to maximize
    def eqn7(p):
        return -1.0*(np.sum(p*p*quadpart + p*linearpart)+5)
    # MUST flatten arrays
#    linearpart = np.reshape(linearpart, (numvars,1))
#    quadpart = np.reshape(quadpart, (numvars,1))
#    initial = np.reshape(initial, (numvars,1))
    bounds = Bounds(initial*0.9, initial*1.1)
    opresult = minimize(eqn7, initial, bounds=bounds)
    newprice = opresult.x
    newprice_scipy = np.reshape(newprice, (numvars,1))
    list_scipy.append(newprice_scipy)
    initial = np.array(newprice_scipy.flatten(), dtype='float')

initial = np.array([1,1], dtype='float')

"""using mosek"""
for i in range(50):
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
    #            bkx = [mosek.boundkey.fr] * numvar
                
                # Bound values for variables
                temppricelow = initial*0.9
                temppricehigh = initial*1.1
                blx = []
                bux = []
    #            blx = [inf] * numvar
    #            bux = [inf] * numvar
                for i in range(numvar):
                    blx.append(temppricelow[i])
                    bux.append(temppricehigh[i])
                
                # Objective linear coefficients
                c = linearpart
                
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
                qval = 2*quadpart
    
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
    newprice_mosek = np.reshape(newprice, (numvars,1))
    list_mosek.append(newprice_mosek)
    initial = np.array(newprice_mosek.flatten(), dtype='float')







