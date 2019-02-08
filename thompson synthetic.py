# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:20:03 2019

@author: Samuel
"""

import numpy as np
np.random.seed(10)
gammastar = np.random.uniform(-3,-1,100)

beta = 0.5
price0 = [12]*100
price0 = np.array(price0)

np.random.seed(10)
f1 = np.random.uniform(0.5,5,100)

np.random.seed(1)
data = np.random.uniform(0.5,5,100)
# each row is 1 observation, each column is 1 product
data = np.reshape(data, (1,100))
np.random.seed(2)
for i in range(99):
    data = np.append(data,np.reshape(np.random.uniform(0.5,5,100),(1,100)), axis=0)

# =============================================================================
# MAX-REV-PASSIVE
# =============================================================================













