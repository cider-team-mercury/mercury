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

n_fe_ol = 0.
n_fe_opx = 0.
ol = olivine(n_fe_ol)
opx = orthopyroxene(n_fe_opx)

fol = 0.2; fopx = 1. - fol
rock = burnman.Composite([fol,fopx],[ol,opx])


# outer core xSmax = 0.162, xSi = .30 
# wSi = .18; wS = 0.0; wFe = 1. - wS -wSi
# xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe + wS/mS)
# xS = (wS/mS) / ( wS/mS + wFe/mFe + wSi/mSi)

xS = .05; xSi = .15 ; xFe = 1. - xS - xSi

liquidFe = iron_liquid() # pure liquid Fe
liquidFeS = ironSulfideLiquid(xS) # solution of liquid Fe and FeS
liquidFeSi = ironSilicideLiquid(xSi) # solution of FeSi and Fe
liquidFeSSi = ironSulfideSilicideLiquid(xS,xSi) # ternary solution

# inner core / FeS layer
DSi = 1.
xSi_solid = DSi * xSi
iron = iron()
solidFeS = iron_sulfide() # solid FeS
solidFeSi = ironSilicideAlloy(xSi_solid) # solid solution of Si in Fe



# Structural Parameters
icb = 1300.0e3
cmb = 2020.0e3
R = 2440.0e3
dFeS = 100 * 1e3

# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = [2250.,1900.,1800.]

# build planet!
# merc = Planet([cmb,R],[fe,ol],['bm3','bm3'])
merc = Planet([icb,cmb,R],[solidFeSi,liquidFeSSi,rock],T0)

# # Integrate!
merc.integrate(n_slices,P0,n_iter=5,plot=False)
print merc.moment_over_mr2()
C = merc.moment_of_inertia_list()
print C[-1] / np.sum(C)

plt.subplot(141)
plt.plot(merc.radial_profile()/1.e3, merc.density_profile())
plt.xlabel(r"Radius [$km$]")
plt.ylabel(r"Density [$kg/m^3$]")

plt.subplot(142)
plt.plot(merc.radial_profile()/1.e3, merc.gravity_profile())
plt.xlabel(r"Radius [$km$]")
plt.ylabel(r"Gravity [$m/s^2$]")

plt.subplot(143)
plt.plot(merc.radial_profile()/1.e3, merc.pressure_profile()/1.e9)
plt.xlabel(r"Radius [$km$]")
plt.ylabel(r"Pressure [$Pa$]")

plt.subplot(144)
plt.plot(merc.radial_profile()/1.e3, merc.temperature_profile())
plt.xlabel(r"Radius [$km$]")
plt.ylabel(r"Temperature [$K$]")


plt.show()
