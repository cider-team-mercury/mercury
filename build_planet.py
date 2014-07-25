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
    """
    Define a planet specified by a number of layers of a given thickness,
        material and temperature.
    """
    def __init__(self,  boundaries, compositions, temperatures, methods=None):
        """
        Parameters
        ----------
        boundaries: list of increasing upper boundary radii of layers in km

        compositions: list of burnman.Composite or burnman.Material describing 
            the material of each layer

        temperatures: temperature of the upper boundary for each layer
        methods: list of EOS fitting method

        Optional
        ----------
        methods: list of burnman EOS methods to be used for each material
            in compositions, default: 'slb'
        """

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
        Find densities for a given set of pressures, temperatures and radii
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
        '''
        Find gravity for a given set of densities and radii
        '''
        rhofunc = UnivariateSpline(radii, density )
        poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
        grav = np.ravel(odeint( poisson, 0.0, radii ))
        grav[1:] = grav[1:]/radii[1:]/radii[1:]
        grav[0] = 0.0
        return grav

    def compute_pressure(self, density, gravity, radii):
        '''
        Find pressure for a given set of densities, gravity and radii
        '''
        depth = radii[-1]-radii
        rhofunc = UnivariateSpline( depth[::-1], density[::-1] )
        gfunc = UnivariateSpline( depth[::-1], gravity[::-1] )
        pressure = np.ravel(odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
        return pressure[::-1]

    def compute_adiabat(self,pressures,radii,T_bound):
        """
        Find temperatures for a given set of pressures, radii and boundary temperatures 
                using adiabatic profiles.
        """

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
        """
        Find temperatures for a given set of pressures, radii and boundary temperatures 
                using isothermal profiles.
        """
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

    def calculate_mass_moments(self,density,radii):
        '''
        Calculate the integrated mass of the body.

        Parameters
        ----------
        density : Array with densities [kg m^-3] at a given radius in planet

        radius : Corresponding radii [m] in planet

        Returns
        ----------
        Mtotal : Integrated mass [kg] of the entire planet

        Mlayer : Integrated mass [kg] of each layer in planet

        Ctotal_norm : Normalized moment [M R^2] of the entire planet

        Clayer_norm : Normalized moment [M R^2] of each layer in planet

        MR2 : Normalization factor for moments [kg m^2] defined as 
                Mtotal * (radius of planet)^2
        '''
        
        Mlayer = np.empty_like(self.boundaries)
        Clayer = np.empty_like(self.boundaries)

        # iterate over layers
        last = 0.
        for i,bound in enumerate(self.boundaries):
            layer =  (radii > last) & ( radii <= bound)
            r2 = radii[ layer ]
            r1= np.hstack([np.array([last]),r2[:-1]])
            rho = density[ layer]

            m = 4. / 3. * np.pi * rho * ( r2**3 - r1**3 )
            
            # http://en.wikipedia.org/wiki/List_of_moments_of_inertia
            # Sphere (shell) of radius r2, with centered spherical cavity of radius r1 and mass m
            c =  2. * m / 5. * ( r2**5 - r1**5 ) / ( r2**3 - r1**3)

            # save mass and moment of the layer
            Mlayer[i] = np.sum(m)
            Clayer[i] = np.sum(c)

            last = r2[-1]

        # sum and normalize the moments
        R = radii[-1]
        Mtotal = np.sum(Mlayer)
        Ctotal = np.sum(Clayer)

        Clayer_norm = Clayer / ( Mtotal * R**2)
        Ctotal_norm = Ctotal / ( Mtotal * R**2)

        MR2 =  Mtotal * R**2

        return Mtotal, Mlayer, Ctotal_norm, Clayer_norm, MR2


    def integrate(self,n_slices,P0,n_iter=5,profile_type='adiabatic',plot=False,verbose=True):
        """
        Iteratively determine the pressure, density temperature and gravity profiles for the
        as a function of radius within a planet.

        Parameters
        ----------
        n_slices : number of radial slices

        P0 : initial guess for central pressure in Pa

        Optional
        ----------
        n_iter : number of iterations (default: 5)

        profile_type : temperature profile type ('adiabatic' or 'isothermal, default: 
                'adiabatic')

        plot : create plot of density, gravity, pressure and temperature as a function
                of radius (default: False)
        
        verbose : (default: True)

        Returns
        ----------
        radius : Array with radii [m] described by n_slices

        density : Array with the density [kg m^-3] calculated for at radius

        gravity : Array with gravitational acceleration [m s^-2] calculated at radius

        pressure : Array with pressure [Pa] calculated at radius

        temperature : Array with temperature [K] calculated at radius
        """
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
            else:
                raise NameError('Invalid profile_type:'+profile_type)
            density = self.evaluate_eos(pressure, temperature, radius)
            gravity = self.compute_gravity(density, radius)
            pressure = self.compute_pressure(density, gravity, radius)

            if plot==True:
                ax1.plot(radius, density)
                ax2.plot(radius, gravity)
                ax3.plot(radius, pressure)
                ax4.plot(radius, temperature)
        
        if plot==True:
            plt.show()
        
        return radius, density, gravity, pressure, temperature
