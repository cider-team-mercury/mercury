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
import burnman.composite as composite

# constants
G = 6.67e-11


class Planet:
    def __init__(self,  boundaries, compositions, methods=None):
        ''' 
        args:
            boundaries: list of increasing outer boundaries of layers
            compositions: list of burnman.Composite of burnman.Material
            methods: list of EOS fitting method
        '''

        for i,c in enumerate(compositions):
            assert( isinstance(c,burnman.Material) )

        for i,b in boundaries:
            assert( i == 0 or b >= boundaries[i-1])

        assert( len(boundaries) == len(compositions)

        self.boundaries = boundaries
        self.compositions = compositions
        self.Nlayer = len(boundaries)

        if methods is None:
            meths = ['slb3'] * self.Nlayer
        else:
            meths = methods

        for m, comp in zip(meths,self.compositions):
            comp.set_method(m)


    def evaluate_eos(self, pressures, temperatures, radii):
        '''
        Find densities for a given set of pressures and temperatures
        args:
            pressures: array of starting pressures in Pa 
            temperatures: array of constant temperatures in K
            radii: array of radii in m
        '''

        assert(radii.max() <= self.boundaries[-1] and radii.min() >= 0. )

        densities = np.empty_like(radii)    

        for i in range(len(radii)):
            if radii[i] > self.cmb:
                density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.ol, np.array([pressures[i]]), np.array([temperatures[i]]))
                densities[i] = density
            else:
                density, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.fe, np.array([pressures[i]]), np.array([temperatures[i]]))
                densities[i] = density

        # iterate over layers
        last = -1.
        for bound,comp in zip(self.boundaries,self.compositions):
            rrange = radii[ last < radii <= bound]
            prange = pressures[last < radii <= bound]
            trange = trange[last < radii <= bound]

            for i in range(len(rrange)):
                    density, vp, vs, vphi, K, G = burnman.velocities_from_rock(comp, np.array([prange[i]]), np.array([trange[i]]))
                    densities[last < radii <= bound] = density

        return densities


    def compute_gravity(self, density, radii):
        rhofunc = UnivariateSpline(radii, density )
        poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
        grav = np.ravel(odeint( poisson, 0.0, radii ))
        grav[1:] = grav[1:]/radii[1:]/radii[1:]
        grav[0] = 0.0
        return grav

    def compute_pressure(self, density, gravity, radii):
        depth = radii[-1]-radii
        rhofunc = UnivariateSpline( depth[::-1], density[::-1] )
        gfunc = UnivariateSpline( depth[::-1], gravity[::-1] )
        pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
        return pressure[::-1]
   
    def integrate(self,n_slices,P0,T0,n_iter=5)
        '''
        Iteratively determine the pressure, temperature and gravity profiles for the
        planet.
        Only Isothermal temperature profile implemented. Should implement adiabatic
        profiles.
        Usage:
            density, gravity, pressures = integrate(n_slices,P0,T0,n_iter=5)
        args:
            n_slices: number of radial slices
            P0: initial central pressure in Pa
            T0: exterior temperature in K
            n_iter: number of iterations (default: 5)
            tol: not implemented
        '''

#         n_slices = 300
        radius = np.linspace(0.e3, self.boundaries[-1], n_slices)
#         pressures = np.linspace(35.0e9, 0.0, n_slices) # initial guess at pressure profile
        pressures = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        temperatures = np.ones_like(pressures)*T0
        gravity = np.empty_like(radius)

        for i in range(n_iter): 
            density = self.evaluate_eos(pressures, temperatures, radius)
            gravity = compute_gravity(density, radius)
            pressures = compute_pressure(density, gravity, radius)

        return density, gravity, pressures


# Material parameters
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

fe = iron()
ol = olivine()

# Structural Parameters
cmb = 2020.0e3
R = 2440.0e3

# integration parameters
n_slices = 300
P0 = 40.0e9
T0 = 1000.

# 
merc = Planet([cmb,R],[iron,ol])

# radius, density, pressure = merc.integrate(n_slices,P0,T0)
# 
# plt.subplot(131)
# plt.plot(radius, density)
# plt.subplot(132)
# plt.plot(radius, gravity)
# plt.subplot(133)
# plt.plot(radius, pressures)
# 
# plt.show()
