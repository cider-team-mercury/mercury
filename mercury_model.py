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
from mercury_materials import *
fe = iron()

n_fe_ol = .05
n_fe_opx = .05
ol = olivine(n_fe_ol)
opx = orthopyroxene(n_fe_ol)

fol = 0.2; fopx = 1. - fol
rock = burnman.Composite([fol,fopx],[ol,opx])


# Structural Parameters
cmb = 2020.0e3
R = 2440.0e3

# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = 0.

# build planet!
# merc = Planet([cmb,R],[fe,ol],['bm3','bm3'])
merc = Planet([cmb,R],[fe,ol])

# # Integrate!
radius, density, gravity, pressure = merc.integrate(n_slices,P0,T0,n_iter=5,plot=True)
