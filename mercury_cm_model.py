'''
mercury_cm_model.py
'''

import numpy as np
import matplotlib.pyplot as plt

import burnman
import burnman.minerals as minerals
import burnman.composite as composite

# from build_cm_planet import cm_planet

from core_partition import partition

# Mass of the layers
from mercury_reference import *

# Fraction of the planets mass in the iron core (inner + outer)
core_Mfrac = 0.9
# Fraction of the cores mass in the inner core
inner_Mfrac = 0.5

M_mantle = M()*(1. - core_Mfrac)
M_inner = M()*core_Mfrac*inner_Mfrac
M_outer = M()*core_Mfrac*(1.-inner_Mfrac)


# Material Properties
from mercury_minerals import *

# molar masses
mFe = 55.845
mSi = 28.0855
mS = 32.066

#mantle minerals
n_fe_ol = 0.0 # iron content of mantle minerals
n_fe_opx = 0.0
ol = olivine(n_fe_ol)
opx = orthopyroxene(n_fe_opx)

# fraction of olivine and orthopyroxene in the mantle
fol = 0.2; fopx = 1. - fol
rock = burnman.Composite([fol,fopx],[ol,opx])

# Total fraction of light elements in the core (in wt. %)
wS = .05
wSi = .05
wFe = 1. - wS - wSi

# Distribution coefficients [Wsolid]/[Wliquid] (Is this correct, the different
# weight percents dont take echother into account).
DS = 0. # DS has to be zero for the current burnman solution model!!!
DSi = 1.0

w_outer, w_inner = partition([wS,wSi],[DS,DSi],inner_Mfrac)

wS_l = w_outer[0]; wSi_l=w_outer[1]; wFe_l = 1.-wS
xS_l = (wS_l/mS) / ( wS_l/mS + wFe_l/mFe + wSi_l/mSi)
xSi_l = (wSi_l/mSi) / ( wS_l/mS + wFe_l/mFe + wSi/mSi)
liquidFeSSi = ironSulfideSilicideLiquid(xS_l,xSi_l) # ternary solution

