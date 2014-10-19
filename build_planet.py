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

        for c in compositions:
            assert( isinstance(c,burnman.Material) )

        for i,b in enumerate(boundaries):
            assert( i == 0 or b >= boundaries[i-1])

        assert( len(boundaries) == len(compositions) )

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

        # iterate over layers
        last = -1.
        for bound,comp in zip(self.boundaries,self.compositions):
            layer =  (radii > last) & ( radii <= bound)
            rrange = radii[ layer ]
            drange = np.empty_like(rrange)
            prange = pressures[ layer ]
            trange = temperatures[ layer ]

            for i in range(len(rrange)):
                    density, vp, vs, vphi, K, G = burnman.velocities_from_rock(comp, np.array([prange[i]]), np.array([trange[i]]))
                    drange[i] = density

            densities[layer] = drange
            last = bound # update last boundary

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
   
    def integrate(self,n_slices,P0,T0,n_iter=5,plot=False):
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

        radius = np.linspace(0.e3, self.boundaries[-1], n_slices)
        pressure = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        temperatures = np.ones_like(pressure)*T0
        gravity = np.empty_like(radius)


        if plot == True:
            f = plt.figure()
            ax1 = plt.subplot(131)
            ax2 = plt.subplot(132)
            ax3 = plt.subplot(133)
            plt.hold(True)

        for i in range(n_iter): 
            density = self.evaluate_eos(pressure, temperatures, radius)
            gravity = self.compute_gravity(density, radius)
            pressure = self.compute_pressure(density, gravity, radius)

            # Plots !
            if plot==True:
                ax1.plot(radius, density)
                ax2.plot(radius, gravity)
                ax3.plot(radius, pressure)
        
        if plot==True:
            plt.show()
    
        return radius, density, gravity, pressure


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
T0 = 0.

# build planet!
# merc = Planet([cmb,R],[fe,ol],['bm3','bm3'])
merc = Planet([cmb,R],[fe,ol])

# # Integrate!
radius, density, gravity, pressure = merc.integrate(n_slices,P0,T0,n_iter=5,plot=True)
 
