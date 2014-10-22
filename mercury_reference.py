'''
Published Mercury Parameters with uncertainty from
    Margot et al. 2012
    Smith et al. 2012
    Zuber et al. 2012
'''

import numpy as np

# constants
G = 6.673e-11

# Margot et al. 2012
def C_over_MR2(): return np.array([.346,.01])
def Cm_over_C(): return np.array([0.432,.025])

# Smith et al. 2012
def GM(): return np.array([22031.78,.02]) 
def M(): return np.array([GM() / G])

# Zuber et al. 2012
def R_equatorial(): return np.array([2439.83,.05])
def R_polar(): return np.array([437.57,.01])
def R_mean(): return np.array([439.59,.05])
