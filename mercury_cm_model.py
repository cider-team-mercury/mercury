'''
mercury_cm_model.py
'''

import numpy as np
import matplotlib.pyplot as plt

import burnman
import burnman.minerals as minerals
import burnman.composite as composite


# from liquidus_model import Solver as Liquidus
from liquidus_model import Solver_no14 as Liquidus

from build_planet_cm import cm_Planet, corePlanet

from core_partition import partition

# Mass of the layers
from mercury_reference import *

# Fraction of the planets mass in the iron core (inner + outer)
core_Mfrac = 0.9
# Fraction of the cores mass in the inner core
inner_Mfrac = 0.5

M_planet = M()[0]
M_mantle = M_planet*(1. - core_Mfrac)
M_inner = M_planet*core_Mfrac*inner_Mfrac
M_outer = M_planet*core_Mfrac*(1.-inner_Mfrac)


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

wS_l = w_outer[0]; wSi_l=w_outer[1]; wFe_l = 1.-wS_l-wSi_l
xS_l = (wS_l/mS) / ( wS_l/mS + wFe_l/mFe + wSi_l/mSi)
xSi_l = (wSi_l/mSi) / ( wS_l/mS + wFe_l/mFe + wSi_l/mSi)
liquidFeSSi = ironSulfideSilicideLiquid(xS_l,xSi_l) # ternary solution

wS_s = w_inner[0]; wSi_s=w_inner[1]; wFe_s = 1.-wS_s-wSi_s
xS_s = (wS_s/mS) / ( wS_s/mS + wFe_s/mFe + wSi_s/mSi)
xSi_s = (wSi_s/mSi) / ( wS_s/mS + wFe_s/mFe + wSi_s/mSi)
assert xS_s == 0. # DS has to be zero for the current burnman solution model!!!
solidFeSi = ironSilicideAlloy(xSi_s) # solid solution of Si in Fe

# find the T(P) liquidus curve for the given wS (is this absurdly slow?)
# could always refit, or add a function to the liquidus model
liq_w = Liquidus()
liquidus = lambda p: liq_w.T_SP(wS_l,p) 


# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = [2200.,1550.,1000.]

# build planet!
merc = corePlanet([M_inner,M_outer,M_mantle],[solidFeSi,liquidFeSSi,rock],T0,
        liquidus=liquidus)

# # Integrate!
merc.integrate(n_slices,P0,n_iter=5,plot=True)
# print merc.moment_over_mr2()


