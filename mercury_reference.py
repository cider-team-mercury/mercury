'''
Published Mercury Parameters with uncertainty from
    Margot et al. 2012
    Smith et al. 2012
    Zuber et al. 2012
'''

import numpy as np

# constants
G = 6.673e-11

# molar masses
mFe = 55.845
mSi = 28.0855
mS = 32.066

# Physical Constraints

# Margot et al. 2012
C_over_MR2 = np.array([.346,.01])
Cm_over_C = np.array([0.432,.025])

# Smith et al. 2012
GM = np.array([2.203178e+13,20000000])
M = GM / G

# Zuber et al. 2012
R_equatorial = np.array([2439.83,.05])*1000.
R_polar = np.array([2437.57,.01])*1000.
R_mean = np.array([2439.59,.05])*1000.

# mantle mineralogy
# iron content of mantle minerals
n_fe_ol = 0.0 
n_fe_opx = 0.0
# fraction of olivine and orthopyroxene in the mantle
fol = 0.2
fopx = 1. - fol

# Iron Alloy Liquid Distribution coefficients [Wsolid]/[Wliquid] 
# (Is this correct, the different weight percents dont take echother into account).
DS = 0. # DS has to be zero for the current  solution model!!!
DSi = 1.0

# Iron Alloy Latent Heat of Fusion
LFe = 247291. # J / kg (Desai 1986 +/- 300 )

# Iron melting temperature from Anzellini
Tm_anzellini = lambda p :  1991.*((p - 5.2e9)/27.39e9 + 1.)**(1./2.38)
