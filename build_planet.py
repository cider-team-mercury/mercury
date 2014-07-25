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
    def __init__(self,  boundaries, compositions, temperatures, methods=None):
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
        self.temperatures = temperatures
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

    def compute_adiabat(self,pressures,radii,T_bound):

        assert( len(T_bound) == self.Nlayer )

        temperatures = np.empty_like(radii)

        # iterate over layers
        last = -1.
        for bound,comp,T0 in zip(self.boundaries,self.compositions,T_bound):
            layer =  (radii > last) & ( radii <= bound)
            rrange = radii[ layer ]
            prange = pressures[ layer ]

            trange = burnman.geotherm.adiabatic(prange[::-1],np.array([T0]),comp)
            temperatures[layer] = trange[::-1]

            last = bound # update last boundary

        return temperatures

    def compute_isotherm(self,radii,T_bound):
        assert( len(T_bound) == self.Nlayer )

        temperatures = np.empty_like(radii)

        # iterate over layers
        last = -1.
        for bound,T0 in zip(self.boundaries,T_bound):
            layer =  (radii > last) & ( radii <= bound)
            rrange = radii[ layer ]

            trange = np.ones_like(rrange)*T0
            temperatures[layer] = trange[::-1]

            last = bound # update last boundary

        return temperatures

    def display_input(self,n_slices,P0,n_iter,profile_type):
        print 'Computing interior structure with layers:\n'
        print 'Planet Model:'
        print '\tNumber of Layers:',self.Nlayer
        print '\tRadii of upper boundaries:',self.boundaries
        print '\tTemperature of upper boundaries,',self.temperatures
        print '\tComposition of boundaries:', self.compositions, '\n'
        print 'Integration parameters:'
        print '\tNumber of radial slices:',n_slices
        print '\tNumber of iterations:',n_iter
        print '\tType of temperature profile:',profile_type
        print '\tIntial guess for central pressure:',P0,'\n'


   
    def integrate(self,n_slices,P0,n_iter=5,profile_type='adiabatic',plot=False,verbose=True):
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

        if verbose:
            self.display_input(n_slices,P0,n_iter,profile_type)

        radius = np.linspace(0.e3, self.boundaries[-1], n_slices)
        pressure = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        temperature = np.ones_like(pressure)*self.temperatures[-1]
        gravity = np.empty_like(radius)

        if plot == True:
            ax1 = plt.subplot(141)
            ax2 = plt.subplot(142)
            ax3 = plt.subplot(143)
            ax4 = plt.subplot(144)
            plt.hold(True)

        for i in range(n_iter): 
            if verbose: print 'Iteration #',i+1

            if profile_type == 'adiabatic':
                temperature = self.compute_adiabat(pressure,radius,self.temperatures)
            elif profile_type == 'isothermal':
                temperature = self.compute_isotherm(radius,self.temperatures)
#                 print temperature
            else:
                raise NameError('Invalid profile_type:'+profile_type)
            density = self.evaluate_eos(pressure, temperature, radius)
            gravity = self.compute_gravity(density, radius)
            pressure = self.compute_pressure(density, gravity, radius)

            # Plots !
            if plot==True:
                ax1.plot(radius, density)
                ax2.plot(radius, gravity)
                ax3.plot(radius, pressure)
                ax4.plot(radius, temperature)
        
        if plot==True:
            plt.show()
    
        return radius, density, gravity, pressure, temperature


 
