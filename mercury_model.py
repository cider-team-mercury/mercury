'''
mercury_model.py
'''

import numpy as np
import matplotlib.pyplot as plt

import burnman
import burnman.minerals as minerals
import burnman.composite as composite

from build_planet import Planet

# Material Properties
from mercury_minerals import *

# molar masses
mFe = 55.845
mSi = 28.0855
mS = 32.066

# fe = iron()

n_fe_ol = 0.5
n_fe_opx = 0.5
ol = olivine(n_fe_ol)
opx = orthopyroxene(n_fe_opx)

fol = 0.2; fopx = 1. - fol
rock = burnman.Composite([fol,fopx],[ol,opx])

# inner core / FeS layer
wSi = 0.15; wFe = 1.-wSi
xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe )
iron = iron()
solidFeS = iron_sulfide() # solid FeS
solidFeSi = ironSilicideAlloy(xSi) # solid solution of Si in Fe

# outer core
wS = 0.05; wFe = 1.-wS
xS = (wS/mS) / ( wS/mS + wFe/mFe )

liquidFe = iron_liquid() # pure liquid Fe
liquidFeS = ironSulfideLiquid(xS) # solution of liquid Fe and FeS
liquidFeSi = ironSilicideLiquid(xSi) # solution of FeSi and Fe
liquidFeSSi = ironSulfideSilicideLiquid(xS,xSi) # ternary solution

# Structural Parameters
icb = 1300.0e3
cmb = 2020.0e3
R = 2440.0e3
dFeS = 100 * 1e3

# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = [2200.,1550.,1500.,1000.]

# build planet!
# merc = Planet([cmb,R],[fe,ol],['bm3','bm3'])
merc = Planet([icb,cmb,cmb+dFeS,R],[iron,liquidFeSSi,solidFeS,rock],T0)

# # Integrate!
radius, density, gravity, pressure, temperature = \
        merc.integrate(n_slices,P0,n_iter=5,plot=True)
#         merc.integrate(n_slices,P0,n_iter=5,plot=True,profile_type='isothermal')

# calculate mass and moments
M, Mlayer, C, Clayer, MR2 = merc.calculate_mass_moments(density,radius)

print '\nMass: ', M,'kg', Mlayer
print 'Moment of inertia:', C,'MR^2', Clayer
print 'Cm / C:',  np.sum(Clayer[-2:]) / C
print 'MR^2 :',MR2, ' kg m^2'
print 'Surface gravity :',gravity[-1],'m s^-2'
