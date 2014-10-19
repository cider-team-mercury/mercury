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
import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline

import burnman
import burnman.minerals as minerals
import burnman.composite as composite

# constants
G = 6.67e-11

class cm_Planet:
    """
    Define a planet specified by a number of layers of a given mass,
        material and temperature.
    """
    def __init__(self,  boundaries, masses, temperatures, methods=None):
        """
        Parameters
        ----------
        masses: list of layer masses ordered in to out

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

#         for i,b in enumerate(boundaries):
#             assert( i == 0 or b >= boundaries[i-1])

        assert( len(masses) == len(compositions) )

        
        self.masses = masses

        self.boundaries = np.zeros_like(masses)
        self.compositions = compositions
        self.boundary_temperatures = temperatures
        self.Nlayer = len(boundaries)

        if methods is None:
            meths = ['slb3'] * self.Nlayer
        else:
            meths = methods

        for m, comp in zip(meths,self.compositions):
            comp.set_method(m)

        self.massBelowBoundary = np.zeros(len(masses)) #integrated mass up to the ith layer
        msum = 0.
        for i,m in enumerate(masses):
            msum += m
            self.massBelowBoundary[i] = msum

        assert self.massBelowBoundary[-1] == np.sum(self.masses)


    def evaluate_eos(self):
        '''
        Find densities for a given set of pressures, temperatures

        This does not require the radii to have been determined yet.
        '''

#         assert(self.radius[-1] == self.boundaries[-1] and self.radius[0] == 0. )


        # iterate over layers
        last = -1.
        for bound,comp in zip(self.massBelowBoundary,self.compositions):
            layer =  (self.int_mass > last) & ( self.int_mass <= bound)
            mrange = self.int_mass[ layer ] #range in int_mass within the layer
            drange = np.empty_like(mrange)
            prange = self.pressure[ layer ]
            trange = self.temperature[ layer ]

            for i in range(len(mrange)):
                    rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(comp, np.array([prange[i]]), np.array([trange[i]]))
                    drange[i] = rho

            # set the self.density within the layer
            self.density[layer] = drange
            last = bound # update last boundary

    def compute_radii(self):
        '''
        Convert from an self.int_mass and self.density to self.radius.

        Note: This uses trapezoidal rule (should probably make more accurate).
        Checked in case of uniform density
        '''
#         assert int_mass[0] >= 0.
        d_plus = self.density
        d_minus = np.hstack((self.density[0],self.density[:-1])) 
        avg_density = (d_plus + d_minus) / 2.

        mass = self. int_mass - np.hstack((0.,self.int_mass[:-1]))

        radii = np.zeros_like(self.int_mass)
        r = 0.
        for i,m in enumerate(mass):
            rnext = (m / avg_density[i] * 3. / 4. / np.pi + r**3.)**(1./3.)
            radii[i] = rnext
            r = rnext

        # set self.radius
        self.radius = radii

    def compute_boundaries(self):
        '''
        Determine the positions of the boundaries in meters.

        self.int_mass, and self.radius should both be updated
        '''

        radfunc = UnivariateSpline(self.int_mass,self.radius)

        # set self.boundaries
        self.boundaries = np.array([ radfunc(m) for m in massBelowBoundary])

    def compute_gravity(self):
        '''
        Find gravity for a given set of densities and radius

        requires self.radius to be computed
        '''
        rhofunc = UnivariateSpline(self.radius, self.density )
        poisson = lambda p, x : 4.0 * np.pi * G * rhofunc(x) * x * x
        grav = np.ravel(integrate.odeint( poisson, 0.0, self.radius ))
        grav[1:] = grav[1:]/self.radius[1:]/self.radius[1:]
        grav[0] = 0.0

        # set self.gravity
        self.gravity = grav

    def compute_pressure(self):
        '''
        Find pressure for a given set of densities, gravity and radius

        requires self.radius to be computed
        '''
        depth = self.radius[-1]-self.radius
        rhofunc = UnivariateSpline( depth[::-1], self.density[::-1] )
        gfunc = UnivariateSpline( depth[::-1], self.gravity[::-1] )
        press = np.ravel(integrate.odeint( (lambda p, x : gfunc(x)* rhofunc(x)), 0.0,depth[::-1]))
        self.pressure = press[::-1]

    def compute_adiabat(self):
        """
        Find temperatures for a given set of pressures, radii and boundary temperatures 
                using adiabatic profiles.

        Does not require self.radius to be set.
        """

        assert( len(self.boundary_temperatures) == self.Nlayer )

        # iterate over layers
        last = -1.
        for bound,comp,T0 in zip(self.massBelowBoundary,self.compositions,self.boundary_temperatures):
            layer =  (self.int_mass > last) & ( self.int_mass <= bound)
            mrange = self.int_mass[ layer ]
            prange = self.pressure[ layer ]

            trange = burnman.geotherm.adiabatic(prange[::-1],np.array([T0]),comp)
            self.temperature[layer] = trange[::-1]
            last = bound # update last boundary

    def compute_isotherm(self):
        """
        Find temperatures for a given set of pressures, radii and boundary temperatures 
                using isothermal profiles.
        """
        assert( len(self.boundary_temperatures) == self.Nlayer )

        # iterate over layers
        last = -1.
        for bound,comp,T0 in zip(self.massBelowBoundary,self.compositions,self.boundary_temperatures):
            layer =  (self.int_mass > last) & ( self.int_mass <= bound)
            mrange = self.int_mass[ layer ]

            trange = np.ones_like(mrange)*T0
            self.temperature[layer] = trange[::-1]

            last = bound # update last boundary

    def display_input(self,n_slices,P0,n_iter,profile_type):
        print 'Computing interior structure with layers:\n'
        print 'Planet Model:'
        print '\tNumber of Layers:',self.Nlayer
        print '\tRadii of upper boundaries:',self.boundaries
        print '\tTemperature of upper boundaries,',self.boundary_temperatures
        print '\tComposition of boundaries:', self.compositions, '\n'
        print 'Integration parameters:'
        print '\tNumber of radial slices:',n_slices
        print '\tNumber of iterations:',n_iter
        print '\tType of temperature profile:',profile_type
        print '\tIntial guess for central pressure:',P0,'\n'


    def integrate(self,n_slices,P0,n_iter=5,profile_type='adiabatic',plot=False,verbose=True):
        """
        Iteratively determine the pressure, density temperature and gravity profiles for the
        as a function of radius within a planet.

        Parameters
        ----------
        n_slices : number of steps in integrated mass

        P0 : initial guess for central pressure in Pa

        Optional
        ----------
        n_iter : number of iterations (default: 5)

        profile_type : temperature profile type ('adiabatic' or 'isothermal, default: 
                'adiabatic')

        plot : create plot of density, gravity, pressure and temperature as a function
                of radius (default: False)
        
        verbose : (default: True)
        """
        if verbose:
            self.display_input(n_slices,P0,n_iter,profile_type)

        self.int_mass = np.linspace(0.,self.massBelowBoundary[-1], n_slices)
        self.pressure = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        # take isothermal starting T profile
        self.temperature = np.ones_like(self.pressure)*self.boundary_temperatures[-1]

        self.radius = np.empty_like(self.int_mass)
        self.boundaries = np.empty_like(self.massBelowBoundary)

        self.gravity = np.empty_like(self.int_mass)
        self.density = np.empty_like(self.int_mass)


        if plot == True:
            ax1 = plt.subplot(141)
            ax2 = plt.subplot(142)
            ax3 = plt.subplot(143)
            ax4 = plt.subplot(144)
            plt.hold(True)

        for i in range(n_iter): 
            # Calculate temperature and density before finding radii.
            if verbose: print 'Iteration #',i+1

            if profile_type == 'adiabatic':
                self.compute_adiabat()
            elif profile_type == 'isothermal':
                self.compute_isotherm()
            else:
                raise NameError('Invalid profile_type:'+profile_type)
            self.evaluate_eos()
            
            # find radii from the calculated density profile.
            self.compute_radii()
            self.compute_boundaries()

            # compute gravity and pressure from radii
            self.compute_gravity()
            self.compute_pressure()

            if plot==True:
                ax1.plot(self.radius, self.density)
                ax2.plot(self.radius, self.gravity)
                ax3.plot(self.radius, self.pressure)
                ax4.plot(self.radius, self.temperature)
        
        if plot==True:
            plt.show()

#     def mass_list(self):
#         '''
#         Returns a list of masses of the planet [kg]
#         '''
#         masses = np.empty( len(self.compositions) )
#         rhofunc = UnivariateSpline(self.radius, self.density )
#         for i,layer in enumerate(self.compositions):
#             masses[i] = integrate.quad( lambda r : 4.0*np.pi*r*r*rhofunc(r) ,
#                                      (0.0 if i==0 else self.boundaries[i-1]), self.boundaries[i] )[0]
#         return masses
  
    def mass(self):
        return self.massBelowBoundary[-1]
  
    def moment_of_inertia_list(self):
        '''
        Returns a list of moments of inertia of the planet [kg m^2]
        '''
        moments = np.empty( len(self.compositions) )
        rhofunc = UnivariateSpline(self.radius, self.density )
        for i,layer in enumerate(self.compositions):
            moments[i] = integrate.quad( lambda r : 8.0/3.0*np.pi*rhofunc(r)*r*r*r*r, 
                             (0.0 if i==0 else self.boundaries[i-1]), self.boundaries[i] )[0]
        return moments

    def moment_of_inertia(self):
        '''
        Returns the total moment of inertia of the planet [kg m^2]
        '''
        return np.sum(self.moment_of_inertia_list())

    def moment_over_mr2(self):
        '''
        Returns the total moment of inertia divided by MR^2
        '''
        return self.moment_of_inertia() / self.mass() / self.radius[-1] /self.radius[-1]
           
 
    def radial_profile(self):
        return self.radius
    def density_profile(self):
        return self.density
    def gravity_profile(self):
        return self.gravity
    def pressure_profile(self):
        return self.pressure
    def temperature_profile(self):
        return self.temperature
    def mass_profile(self):
        return self.int_mass
