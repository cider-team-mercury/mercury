"""
Given a model for different homogeneous layers, 
and an equation of state, we would like to 
come up with density-pressure-gravity curves 
for the planet.  We need to solve Poisson's
equation for gravity, The hydrostatic equation 
for pressure, and the equation of state for 
density.  These are all interrelated, so we 
need so solve them iteratively.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import burnman
import burnman.minerals as minerals


G = 6.67e-11


class iron (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 6.6e-6,
            'K_0': 180.0e9,
            'Kprime_0': 4.9,
            'G_0': 130.9e9,
            'Gprime_0': 1.92,
            'molar_mass': .0558,
            'n': 1,
            'Debye_0': 300.,
            'grueneisen_0': 1.5,
            'q_0': 1.5,
            'eta_s_0': 2.3 }

class olivine (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 11.24e-6,
            'K_0': 161.0e9,
            'Kprime_0': 3.9,
            'G_0': 130.9e9,
            'Gprime_0': 1.92,
            'molar_mass': .0403,
            'n': 2,
            'Debye_0': 773.,
            'grueneisen_0': 1.5,
            'q_0': 1.5,
            'eta_s_0': 2.3 }

class Mercury:
  def __init__(self, cmb):
    self.cmb = cmb
    self.ol = olivine()
    self.fe = iron()
    self.ol.set_method('slb3')
    self.fe.set_method('slb3')

  def evaluate_eos(self, pressures, temperatures, radii):
    densities = np.empty_like(radii)    

    for i in range(len(radii)):
      if radii[i] > self.cmb:
        density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.ol, np.array([pressures[i]]), np.array([temperatures[i]]))
        densities[i] = density
      else:
        density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.fe, np.array([pressures[i]]), np.array([temperatures[i]]))
        densities[i] = density
    return densities


def compute_gravity(density, radii):
  rhofunc = UnivariateSpline(radii, density )
  poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
  grav = np.ravel(odeint( poisson, 0.0, radii ))
  grav[1:] = grav[1:]/radii[1:]/radii[1:]
  grav[0] = 0.0
  return grav

def compute_pressure(density, gravity, radii):
  depth = radii[-1]-radii
  rhofunc = UnivariateSpline( depth[::-1], density[::-1] )
  gfunc = UnivariateSpline( depth[::-1], gravity[::-1] )
  pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
  return pressure[::-1]
   
  


n_slices = 300
radius = np.linspace(0.e3, 2440.e3, n_slices)
pressures = np.linspace(35.0e9, 0.0, n_slices) # initial guess at pressure profile
temperatures = np.ones_like(pressures)*0.0
gravity = np.empty_like(radius)


merc = Mercury(2020.0e3)


for i in range(5):
  density = merc.evaluate_eos(pressures, temperatures, radius)
  gravity = compute_gravity(density, radius)
  pressures = compute_pressure(density, gravity, radius)

  plt.subplot(131)
  plt.plot(radius, density)
  plt.subplot(132)
  plt.plot(radius, gravity)
  plt.subplot(133)
  plt.plot(radius, pressures)

plt.show()
