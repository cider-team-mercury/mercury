'''
mercury_model.py
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


# Structural Parameters
icb = 1500.0e3
cmb = 2020.0e3
R = 2440.0e3

# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = [2000.,1850.,1800.] # approximate temperatures from tosi

n_fe_ol = 0.0
n_fe_opx = 0.0

DSi = 1.
xSi_arr = np.linspace(0.,0.3,7)
xS_arr = np.linspace(0.,0.15,4)

for xSi in xSi_arr:
    for xS in xS_arr:

ol = olivine(n_fe_ol)
opx = orthopyroxene(n_fe_opx)

fol = 0.2; fopx = 1. - fol
rock = burnman.Composite([fol,fopx],[ol,opx])

# outer core xSmax = 0.162, xSi = .30 
wSi = .18; wS = 0.0; wFe = 1. - wS -wSi
xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe + wS/mS)
xS = (wS/mS) / ( wS/mS + wFe/mFe + wSi/mSi)

liquidFeSSi = ironSulfideSilicideLiquid(xS,xSi) # ternary solution

# inner core / FeS layer
DSi = 1.
xSi_solid = DSi * xSi
solidFeS = iron_sulfide() # solid FeS
solidFeSi = ironSilicideAlloy() # solid solution of Si in Fe




# build planet!
# merc = Planet([cmb,R],[fe,ol],['bm3','bm3'])
merc = Planet([icb,cmb,cmb+dFeS,R],[iron,liquidFeSSi,solidFeS,rock],T0)

# # Integrate!
merc.integrate(n_slices,P0,n_iter=5,plot=False)
print merc.moment_over_mr2()

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
