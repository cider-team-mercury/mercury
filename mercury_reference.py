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
C_over_MR2 = np.array([.346,.01])
Cm_over_C = np.array([0.432,.025])

# Smith et al. 2012
GM = np.array([2.203178e+13,20000000])
M = GM / G

# Zuber et al. 2012
R_equatorial = np.array([2439.83,.05])
R_polar = np.array([437.57,.01])
R_mean = np.array([439.59,.05])
