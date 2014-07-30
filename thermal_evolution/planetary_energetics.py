import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

from scipy.integrate import odeint

class Layer(object):
    '''
    The layer base class defines the geometry of a spherical shell within
    a planet.
    '''

    def __init__( self, inner_radius, outer_radius, params={}):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.thickness = outer_radius-inner_radius

        assert( self.thickness > 0.0 )

        self.inner_surface_area = 4.0 * np.pi * self.inner_radius**2.
        self.outer_surface_area = 4.0 * np.pi * self.outer_radius**2.

        self.volume = 4.0/3.0 * np.pi * ( self.outer_radius**3. - self.inner_radius**3.)

        try:
            self.calculate_mass()
        except:
            pass

        self.params = params
        self.__dict__.update(params)

    def set_params(self,params):
        '''
        Set material parameters for the layer 
        '''
        self.params.update(params)
        self.__dict__.update(params)

    def calculate_mass(self):
        assert( 'rho' in self.params)
        self.mass = self.volume * self.params['rho']

    def set_average_temperature(self,T):
        '''
        Set the average temperature for the layer
        '''
        self.T_a = T

    def set_boundary_temperatures(self,outer_temperature,inner_temperature): 
        '''
        All layers should be able to track the temperatures of the their outer and inner
        boundary.
        '''
        self.outer_temperature = outer_temperature
        self.inner_temperature = inner_temperature

    def lower_heat_flux (self):
        raise NotImplementedError("Need to define a heat flux function")

    def upper_heat_flux (self):
        raise NotImplementedError("Need to define a heat flux function")
   
    def update( self, lower_temperature, upper_temperature, time ):
        raise NotImplementedError("Need to define a physics yo")

    def radiogenic(self,t):
        assert( 'H_0' in self.params and 'lambda' in self.params)
        lam = self.params['decay_constant']
        H_0 = self.params['H_0']

        return H_0 * np. exp( -lam * t )

class Planet(object):

    def __init__( self, layers, T0 ):
       self.layers = layers
       self.temperatures = T0
       self.Nlayers = len(layers)
#        self.time = 0. # time

       self.radius = self.layers[-1].outer_radius 
       self.volume = 4./3. * np.pi * self.radius**3

       for T,layer in zip(T0,self.layers):
           layer.set_average_temperature(T)

    def integrate( self ):
        raise NotImplementedError("Need to define a physics yo")
       
    def draw(self):

        c = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc', \
                '#e5d8bd','#fddaec','#f2f2f2']
        fig = plt.figure()
        axes = fig.add_subplot(111)

        wedges = []
        for i,layer in enumerate(self.layers):
           wedges.append( patches.Wedge( (0.0,0.0), layer.outer_radius, 70.0, 110.0,\
                   width=layer.thickness, color=c[i]) )
        p = PatchCollection( wedges, match_original = True )
        axes.add_collection( p )
        r = max( [l.outer_radius for l in self.layers ] ) * 1.1

        axes.set_ylim( 0, r)
        axes.set_xlim( -r/2.0 , r/2.0 )
        plt.axis('off')
        plt.show()

        def convectionODE(self,t,T_a,T_surface = 0.):

            T_out =  np.array([ layer.T_outer(T) for layer in self.layers \
                    for T in T_a] )
            T_in = np.array( [ layer.T_inner(T) for layer in self.layers \
                    for T in T_a] )

            theta = np.array([ layer.
            # determine the boundary conditions
            T_bound = np.empty_like(T_out)
            T_bound[-1] = T_surface

            # estimate the boundary layer temperatures
            T_bound_up = np.empty_like(T_bound)
            T_bound_up[-1] = (T_out[-1] + T_in[-1]) / 2.

            T_bound_low = np.enpty_like(T_bound)
            T_bound_low[0] = T_in[0]
            T_bound[0] = T_in[0]        #this doesn't make any sense
                                        #made-up temperature at the center of the Earth

            print 'Setting T_bound_up, T_bound_low'

            for i in np.arange(Nlayer-1,0,-1)-1:
                T_bound_up[i] = 0.75 * T_out[i] + 0.25 * T_in(i+1)
                T_bound_low[i+1] = 0.25 * T_out[i] + 0.75 * T_in[i+1]



class ConvectiveLayer(Layer):
    '''
    Possible regimes 'symmetric','transitional','stagnant_lid'
    '''
    def __init__(self,inner_radius,outer_radius):
        params = {
            'regime' : 'isoviscous',
            'mu' : 1.0,  #conversion factor in each layer (vector); STO table 13.2
            'Ra_c' : 2e3, #critical Rayleigh number
#             eta_0 : 0.9e20, #viscosity Pa.s
            'rho' : 3.0e3, # density; kg/m^3
            'H_0' : 0., #heat production W/m^3
            'decay_constant' : 1.38e-17, # some half-life
            'nu_0' : 1.65e2, #kinematic viscosity m^2/s
            'A_0' : 7.e4,
            'k' : 3., #thermal conductivity W/m/K
            'kappa' : 1.e-6, #thermal diffusivity m^2/s
            'alpha' : 2.e-5, #thermal expansion 1/K
            'g' : 3., #acceleration of gravity m/s^2
            'beta' : 1./3, #parameter Nu=Ra^(1/beta)
            } 
            
        Layer.__init__(self,inner_radius,outer_radius,params)

        params['Cp'] = params['k'] / ( params['rho'] * params['kappa'] )
        params['gamma'] = 1. / params['beta']
        self.calculate_mass()


# I would like to add these but they may be incosistant with definition of T_a

#     def calculate_rayleigh(self,deltaT):
#         return self.g * self.alpha * self.thickness ** 3 * deltaT \
#                 / self.kappa / viscosity(self.T_a)

    def T_outer(self,T_a):
        self.outer_temperature = T_a / self.mu
        return T_a / self.mu

    def T_inner(self,T_a):
        self.inner_temperature =  2 * T_a - self.T_outer(T_a)
        return 2 * T_a - self.T_outer(T_a)

    def viscosity(self,T): # should T be self.T_a
        '''
        Temperature dependent viscosity using 'nu_0' and 'A_0'.

        If no A_0 provided: isoviscous with nu = nu_0.
        '''
        
        nu_0 = self.params['nu_0']

        if 'A_0' in self.params:
            A_0 = self.params['A_0']

        return nu_0 * np.exp(A_0 / T)


    def theta(self, T_a): # can we give this a better name?
        '''
        Temperature drop to heat flux conversion factor
        Schubert (13.2.9)
        ''' 

        k = self.params['k']
        alpha = self.params['alpha']
        g = self.params['g']
        kappa = self.params['kappa']
        Ra_c = self.params['Ra_c']
        beta = self.params['beta']
        d = self.thickness

        visc = self.viscosity(T_a)

        # note the addition of d**3 to make the expression in the power unitless
        return k *  (alpha * g * d**3./ ( kappa * visc * Ra_c))**beta

#     function dTadt=convectionODE(t,Ta,Tuf,Tlf,Ts,Theta,gamma,A,M,H,C,n);
    
class CoreLayer(Layer):
    def __init__(self,inner_radius,outer_radius):
        params = {
            'regime' : 'isoviscous',
            'eta' : 1.1,
            'mu' : 1.,  #conversion factor in each layer (vector); STO table 13.2
            'rho' : 8.0e3, # density; kg/m^3
            'g' : 3.8, #acceleration of gravity m/s^2
            'L' : 2.5e5, # latent_heat
            'Cp' : 450.,
            'Eg' :5.e5
            } 
            
        Layer.__init__(self,inner_radius,outer_radius,params)

        self.calculate_mass()

    def adiabat(self,p):
        pass

    def pressure(self,r,x):
        pressure

    def liquidus(self,p,x):

    def set_T_upper(

T0 = [ 2000., 1000.]        
core = CoreLayer(0.0,2020.0e3)
mantle = ConvectiveLayer(2020.e3, 2440.0e3)

mercury = Planet( [core,mantle], T0)
       
# mercury.draw()

    
