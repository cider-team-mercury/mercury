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
from scipy.misc import derivative

import burnman
import burnman.minerals as minerals
import burnman.composite as composite


# constants
G = 6.67e-11

class cm_Planet(object):
    """
    Define a planet specified by a number of layers of a given mass,
        material and temperature.
    """
    def __init__(self,  masses, compositions, temperatures, methods=None):
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
        self.Nlayer = len(masses)

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
            vprange =  np.empty_like(mrange)
            vsrange =  np.empty_like(mrange)
            vphirange = np.empty_like(mrange)
            Krange =  np.empty_like(mrange)
            Grange = np.empty_like(mrange)

            prange = self.pressure[ layer ]
            trange = self.temperature[ layer ]

            for i in range(len(mrange)):
                    rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(comp, np.array([prange[i]]), np.array([trange[i]]))
                    drange[i] = rho
                    vprange[i] = vp
                    vsrange[i] = vs
                    vphirange[i] = vphi
                    Krange[i] = K
                    Grange[i] = G
#                     print rho, vp, vs, vphi, K, G

            # set the self.density within the layer
            self.density[layer] = drange
            self.vp[layer] = vprange
            self.vs[layer] = vsrange
            self.vphi[layer] = vphirange
            self.K[layer] = Krange
            self.G[layer] = Grange

            last = bound # update last boundary

    def compute_radii(self):
        '''
        Convert from an self.int_mass and self.density to self.radius.
        '''
        rhofunc = UnivariateSpline(self.int_mass, self.density )

        deltaV = lambda y,m: 1./rhofunc(m)
        volume = np.ravel(integrate.odeint( deltaV, 0.0, self.int_mass ))

        # set self.radius
        self.radius = ( 3. / 4. / np.pi * volume ) ** (1./3.)
    
    def compute_boundaries(self):
        '''
        Determine the positions of the boundaries in meters.

        self.int_mass, and self.radius should both be updated
        '''

        radfunc = UnivariateSpline(self.int_mass,self.radius)

        # set self.boundaries
        self.boundaries = np.array([ radfunc(m) for m in self.massBelowBoundary])

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

    def compute_isotherm_layer(self,idx,T0,fromLowerBound=False):
        '''
        Calculates iosothermal temperature gradient across a layer.
        '''

        layer = self.get_layer(idx)

        mrange = self.int_mass[ layer ]
        prange = self.pressure[ layer ]

        # if fromLowerbounds: # This doesn't matter for an isotherm given T0
        trange = np.ones_like(mrange)*T0
        self.temperature[layer] = trange[::-1]
        print self.temperature[layer]


    def compute_adiabat_layer(self,idx,T0,fromLowerBound=False):
        '''
        Calculates an adiabatic temperature gradient across a layer. Defaults 
        to calculating from the upper boundary downwards. fromLowerBounds=True
        is for calculating up from the lower boundary.
        '''

        comp = self.compositions[idx]
        layer = self.get_layer(idx)

        mrange = self.int_mass[ layer ]
        prange = self.pressure[ layer ]

        if not fromLowerBound:
            trange = burnman.geotherm.adiabatic(prange[::-1],np.array([T0]),comp)
            self.temperature[layer] = trange[::-1]
        else:
            trange = burnman.geotherm.adiabatic(prange,np.array([T0]),comp)
            self.temperature[layer] = trange

    def compute_Eg(self):
        '''
        Directly compute gravitational energy from the current profiles
        '''

        r_func = UnivariateSpline(self.int_mass,self.radius)
        
        dE_dm = lambda y, m: - G * m / r_func(m)

#         Eg = np.ravel(integrate.odeint( dE_dm, 0.0, self.masses[-1] ))
        Eg = np.ravel(integrate.odeint( dE_dm, 0.0, self.int_mass ))

        # save the total gravitational energy
        self.Eg = Eg
        self.gravitational_energy = Eg[-1]

        # save the gravitational energy over the core alone
#         self.core_gravitational_energy = Eg[self.cmb()]

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


    def integrate(self,n_slices,P0,n_iter=5,profile_type='adiabatic',plot=False,
            verbose=True):
        """
        Iteratively determine the pressure, density temperature and gravity profiles
        for the planet as a function of radius within a planet.

        Parameters
        ----------
        n_slices : number of steps in integrated mass

        P0 : initial guess for central pressure in Pa

        Optional
        ----------
        n_iter : number of iterations (default: 5)

        profile_type : temperature profile type ('adiabatic' or 'isothermal,
            default: 'adiabatic')

        plot : create plot of density, gravity, pressure and temperature as a 
            function of radius (default: False)
        
        verbose : (default: True)
        """
        if verbose:
            self.display_input(n_slices,P0,n_iter,profile_type)

        self.int_mass = np.linspace(0.,self.massBelowBoundary[-1], n_slices)
        self.pressure = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        # take isothermal starting T profile
        self.temperature = np.ones_like(self.pressure)*self.boundary_temperatures[-1]

        self.radius = np.zeros_like(self.int_mass)
        self.boundaries = np.zeros_like(self.massBelowBoundary)

        self.gravity = np.zeros_like(self.int_mass)

        # eos parameters
        self.density = np.zeros_like(self.int_mass)
        self.vp = np.zeros_like(self.int_mass)
        self.vs = np.zeros_like(self.int_mass)
        self.vphi = np.zeros_like(self.int_mass)
        self.K = np.zeros_like(self.int_mass)
        self.G = np.zeros_like(self.int_mass)


        if plot == True:
            ax1 = plt.subplot(141)
            ax2 = plt.subplot(142)
            ax3 = plt.subplot(143)
            ax4 = plt.subplot(144)
            plt.hold(True)

        for i in range(n_iter): 

            print 'Initial'
            self.print_state()
            # Calculate temperature and density before finding radii.
            if verbose: print 'Iteration #',i+1

            if profile_type == 'adiabatic':
                self.compute_adiabat()
            elif profile_type == 'isothermal':
                self.compute_isotherm()
            else:
                raise NameError('Invalid profile_type:'+profile_type)
            print 'compute_adiabatic'
            self.print_state()

            self.evaluate_eos()
            print 'evaluate_eos'
            self.print_state()
            
            # find radii from the calculated density profile.
            self.compute_radii()
            print 'compute_radii'
            self.print_state()

            self.compute_boundaries()
            print 'compute_boundaries'
            self.print_state()


            # compute gravity and pressure from radii
            self.compute_gravity()
            print 'compute_gravity'
            self.print_state()

            self.compute_pressure()
            print 'compute_pressure'
            self.print_state()

            if plot==True:
                ax1.plot(self.radius, self.density)
                ax2.plot(self.radius, self.gravity)
                ax3.plot(self.radius, self.pressure)
                ax4.plot(self.radius, self.temperature)
        
        if plot==True:
            ax1.legend()
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
           
    def get_layer(self,idx):
        '''
        Return an index range correpsonding to a single layer.
        '''
        ubound = self.massBelowBoundary[idx]
        if idx == 0:
            lbound = -1.
        else:
            lbound = self.massBelowBoundary[idx-1]

        layer = (self.int_mass > lbound) & (self.int_mass <= ubound)
        return layer

    # access functions for all profiles (this is kind of redundant)
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
    def vp_profile(self):
        return self.vp
    def vs_profile(self):
        return self.vs
    def vphi_profile(self):
        return self.vphi
    def K_profile(self):
        return self.K
    def G_profile(self):
        return self.G


class corePlanet(cm_Planet):
    def __init__(self,  masses, compositions, temperatures, liquidus=None,**kwargs):
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
        liquidus: A function describing the icb temperature (for a given 
        composition).

        methods: list of burnman EOS methods to be used for each material
            in compositions, default: 'slb'
        """
        super(corePlanet,self).__init__(masses, compositions, temperatures,**kwargs)

        # liquidus model
        #Hack because the liquidus_model only works for a single value.
        self.liquidus_at_P = liquidus  

        # Make sure the number of layers is consistent with having a growing core
        assert self.Nlayer >= 3

    def liquidus(self,pressure):
        '''
        Hack because the liquidus_model only works for a single value.
        '''
        try:
            return np.array([ self.liquidus_at_P(p) for p in pressure ])
        except:
            return self.liquidus_at_P(pressure)


    def find_icb_temp(self,idx=0):
        '''
        Use an FeS liquidus model to find a thermodynamically consisten temperature
        for the icb. Assumes that the 0th index refers to the inner core and inner
        core boundary.
        '''
        assert not self.liquidus is None

        m_inner = self.massBelowBoundary[idx]
        p_func = UnivariateSpline(self.int_mass, self.pressure) 
        p_icb = p_func(m_inner)
        t_icb = self.liquidus(p_icb)

#         print 'liquidus:',p_icb,t_icb
        self.temperature[idx] = t_icb
        return t_icb
        
    def compute_temperature(self,inner_isotherm=False,outer_isotherm=False,
            mantle_isotherm=False):
        '''
        Calculate a core adiabat consistent with the size of the inner core
        using the model FeS liquidus.

        Defaults to calculating adiabatic profiles, starting at the icb for
        both inner and outer core.
        '''

        t_icb = self.find_icb_temp()

        # compute inner_core
        if not inner_isotherm:
            self.compute_adiabat_layer(0,t_icb)
        else:
            self.compute_isotherm_layer(0,t_icb)
        
        if not outer_isotherm:
            self.compute_adiabat_layer(1,t_icb,fromLowerBound=True)
        else:
            self.compute_isotherm_layer(1,t_icb,fromLowerBound=True)

        if not mantle_isotherm:
            for i in range(2,self.Nlayer):
                self.compute_adiabat_layer(i,self.boundary_temperatures[i])

        else:
            for i in range(2,self.Nlayer):
                self.compute_isotherm_layer(i,self.boundary_temperatures[i])

        self.boundary_temperatures[0] = t_icb

    def inner_core(self):
        return self.get_layer(0)
    def outer_core(self):
        return self.get_layer(1)
    def mantle(self):
        return -self.inner_core() - self.outer_core()

    def icb(self):
        x = len(self.int_mass) 
#         layer = np.linspace(0,x-1,x)[self.inner_core()]
        layer = np.arange(x)[self.inner_core()]
        return layer[-1]
        
    def cmb(self):
        x = len(self.int_mass) 
#         layer = np.linspace(0,x-1,x)[self.outer_core()]
        layer = np.arange(x)[self.outer_core()]
        return layer[-1]

    def print_state(self,i=150):
        '''
        For debugging
        '''
        print self.int_mass[i],self.radius[i],self.density[i],self.gravity[i],
        print self.pressure[i],self.temperature[i]

    def compute_Eg_over_r(self):
        '''
        Calculate the Eg per volume change in the core
        
        Delta Eg = int ( [ (rho_l - rho_s)*g*4pi*r^2 ] dr )

        Returns the bracketed parameter.
        '''

        # Parameters at the inner core boundary
        idx_icb = self.icb()
        p_icb = self.pressure[idx_icb]
        rho_icb = self.density[idx_icb]
        g_icb = self.gravity[idx_icb]
        r_icb = self.radius[idx_icb]
        t_icb = self.temperature[idx_icb]

        rho_s,vp, vs, vphi, K, G =  burnman.velocities_from_rock(self.compositions[0],
                np.array([p_icb]), np.array([t_icb]) )

        rho_l, vp, vs, vphi, K, G = burnman.velocities_from_rock(self.compositions[1],
                np.array([p_icb]), np.array([t_icb]) )

        # Save Eg / dr
        self.Eg_over_r = (rho_l - rho_s) * g_icb * 4. * np.pi * r_icb**2.

    def compute_Eg(self):
        '''
        Directly compute gravitational energy from the current profiles
        '''

        super(corePlanet,self).compute_Eg()

        # save the gravitational energy over the core alone
        self.core_gravitational_energy = self.Eg[self.cmb()]

    def detect_snow(self):
        '''
        Test whether points in the liquid outer core are above the liquidus.

        This is a very simple check and doesn't consider how liquidus should
        perturb the adiabat
        '''
        p_oc = self.pressure[self.outer_core()]
        t_oc = self.temperature[self.outer_core()]

        t_liq = self.liquidus(p_oc)
        
        # Boolean. True if temperature is below the liquidus (snowing)
        snow = t_liq > t_oc
        self.has_snow = snow.any()

        return snow

    def adiabat_steeper(self):
        '''
        Checks whether the adiabat is steeper than the liquidus. Ultimately this
        should be part of determining the adiabat in the snow regime.
        '''

        p_oc = self.pressure[self.outer_core()]
        t_oc = self.temperature[self.outer_core()]

        t_func = UnivariateSpline(p_oc,t_oc)

        return derivative(t_func,p_oc) > derivative(self.liquidus,p_oc)
        


    def integrate(self,n_slices,P0,n_iter=5,profile_type='adiabatic',plot=False,
            verbose=True):
        """
        Iteratively determine the pressure, density temperature and gravity profiles
        for the planet as a function of radius within a planet, with a consistent
        temperature and pressure for the inner core boundary.

        Sets pressure, temperature, radius, boundaries, gravity, and density, for 
        given profile in integrated mass (int_mass).

        Also sets differences for quantities between the last two iterations 
        for convergence analysis.

        Parameters
        ----------
        n_slices : number of steps in integrated mass

        P0 : initial guess for central pressure in Pa

        Optional
        ----------
        n_iter : number of iterations (default: 5)

        profile_type : temperature profile type ('adiabatic' or 'isothermal,
            default: 'adiabatic')

        plot : create plot of density, gravity, pressure and temperature as a 
            function of radius (default: False)
        
        verbose : (default: True)
        """
        if verbose:
            self.display_input(n_slices,P0,n_iter,profile_type)

        self.int_mass = np.linspace(0.,self.massBelowBoundary[-1], n_slices)
        self.pressure = np.linspace(P0, 0.0, n_slices) # initial guess at pressure profile
        # take isothermal starting T profile
        self.temperature = np.ones_like(self.pressure)*self.boundary_temperatures[-1]

        self.radius = np.zeros_like(self.int_mass)
        self.boundaries = np.zeros_like(self.massBelowBoundary)

        self.gravity = np.zeros_like(self.int_mass)

        # eos parameters
        self.density = np.zeros_like(self.int_mass)
        self.vp = np.zeros_like(self.int_mass)
        self.vs = np.zeros_like(self.int_mass)
        self.vphi = np.zeros_like(self.int_mass)
        self.K = np.zeros_like(self.int_mass)
        self.G = np.zeros_like(self.int_mass)

        if plot == True:
            ax1 = plt.subplot(141);ax1.set_title('rho')
            ax2 = plt.subplot(142);ax2.set_title('g')
            ax3 = plt.subplot(143);ax3.set_title('P')
            ax4 = plt.subplot(144);ax4.set_title('T')
            plt.hold(True)

        for i in range(n_iter): 

            # Keep track of the previous iteration for an idea of the uncertainty
            self.last_state = np.vstack((self.int_mass.copy(),self.radius.copy(),
                self.pressure.copy(), self.temperature.copy(),self.gravity.copy(),
                self.density.copy()) )
            self.last_boundaries = self.boundaries.copy()
            self.last_icb_temp = self.boundary_temperatures[0]

            if verbose: print 'Initial'; self.print_state()

            # Calculate temperature and density before finding radii.
            if verbose: print 'Iteration #',i+1

            # calculate temperature profile with consistent ICB temp
            if profile_type == 'adiabatic':
                self.compute_temperature()
            elif profile_type == 'isothermal':
                self.compute_temperature(inner_isotherm=False,outer_isotherm=False,
                        mantle_isotherm=False)
            else:
                 raise NameError('Invalid profile_type:'+profile_type)

            if verbose: print 'compute_temperature';self.print_state()

            self.evaluate_eos()
            if verbose: print 'evaluate_eos';self.print_state()
            
            # find radii from the calculated density profile.
            self.compute_radii()
            if verbose: print 'compute_radii';self.print_state()

            self.compute_boundaries()
            if verbose: print 'compute_boundaries';self.print_state()

            # compute gravity and pressure from radii
            self.compute_gravity()
            if verbose: print 'compute_gravity'; self.print_state()

            self.compute_pressure()
            if verbose: print 'compute_pressure';self.print_state()

            if plot==True:
                ax1.plot(self.radius, self.density)
                ax2.plot(self.radius, self.gravity)
                ax3.plot(self.radius, self.pressure)
                ax4.plot(self.radius, self.temperature)
        
        # compare differences between last two iterations
        present_state = np.vstack((self.int_mass,self.radius,self.pressure,
                    self.temperature,self.gravity,self.density))

        self.diff_state = present_state - self.last_state
        self.diff_mean = np.mean(self.diff_state,axis=1)
        self.diff_max = max_magnitude(self.diff_state,axis=1)
        self.diff_bounds = self.boundaries - self.last_boundaries
        self.diff_icb_temp = self.boundary_temperatures[0] - self.last_icb_temp

        # Check if snow encountered
        self.detect_snow()
        self.adiabat_steeper()

        # compute quantaties for energy/entropy budget
        self.compute_Eg()
        self.compute_Eg_over_r()

        # print diagnostiscs for last iteration
        if verbose:
            print 'Change during last iteration:'
            print 'mean: ', self.diff_mean
            print 'max: ', self.diff_max
            print 'boundaries: ', self.diff_bounds
            print 'ICB temp: ', self.diff_icb_temp

        if plot==True:
            plt.show()



def max_magnitude(x,**kwargs):
    '''
    Return values with the largest magnitude (+ or -).
    '''
    maxCol = np.argmax(np.abs(x),**kwargs)
    soln = np.zeros_like(maxCol).astype(float)
    for row,col in enumerate(maxCol):
#         print row,col
        soln[row] = x[row,col]
    return soln
