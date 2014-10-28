import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

from scipy.integrate import odeint
from scipy.optimize import minimize_scalar
from define_physics import *

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



class CoreLayer(Layer):
    def __init__(self,inner_radius,outer_radius,params=core_params):
        Layer.__init__(self,inner_radius,outer_radius,params)
        '''
        Note that the default params are loaded from the file "define_physics"
        '''
        # - Parameters from Stevenson et al 1983 for liquiudus and Adiabat
        '''
        Hard Code Adiabat and Liquidus
        parameters from tables (II) (VI) in Stevenson et al 1983 
        '''
        self.stevenson = {
            'alpha_c'   = 2,
            'g'         = 3.8,
            'Tm0'       = 1880.0,
            'Tm1'       = 1.36,
            'Tm2'       = -6.2,
            'Ta1'       = 8.0,
            'Ta2'       = -3.9,
            'x0'        = 0.01,
            'Pcm'       = 10.0,
            'Pc'        = 40.0
            }
        self.calculate_mass()
        self.light_alloy = self.stevenson['x0']


    # - Should write this so we can choose different models, for example
    # - when we initiate CoreLayer we should choose "core_evolution_model = 'stevenson' "
    # - will update with values from Fei et al 1997, 2000
    # - Then we can use the look up tables from Sean as well
    
    def set_light_alloy_concentration(self):
        '''
        Equation (7) from Stevenson 1983
        '''
        x0 = self.stevenson['x0']
        Rc = self.outer_radius
        Ri = self.inner_radius
        self.light_alloy = x0*(Rc**3)/(Rc**3-Ri**3)
        return self.light_alloy

    def set_inner_core_radius(self,Ri):
        self.inner_radius = Ri
        return Ri

    ### We could code the integrals here. 
    def core_mantle_boundary_temp(self):
        return  self.T_average / self.params['mu']

    def stevenson_liquidus(self,Pio,p=self.stevenson):
        '''
        Equation (3) from Stevenson 1983
        
        Calculates the liquidus temp for a given pressure at the inner core
        outer core boundary Pio
        '''
        x = self.light_alloy
        a  = p['alpha_c']
        c1 = p['Tm0']
        c2 = p['Tm1']
        c2 = p['Tm2']
        return c1*(1-a*x)*(1 + c2*Pio +c3*Pio**2)        
    
    def stevenson_adiabat(self,Pio,p=self.stevenson):
        '''
        Equation (4) from Stevenson 1983

        Calculates adiabat temp for a given pressure at the inner core 
        outer core boundary Pio
        '''
        Tcm = self.core_mantle_boundary_temp()
        c1  = p['Ta1']
        c2  = p['Ta2']
        Pcm = p['Pcm']
        return Tcm*(1+c1*Pio+c2*Pio**2)/(1+c1*Pcm+c2*Pcm**2)
    
    def calculate_pressure_io_boundary(self):
        Pcm = self.stevenson['Pcm']
        Pc  = self.stevenson['Pc']
        diff_function = lambda Pio: np.abs(self.stevenson_adiabat(Pio)-self.stevenson_liquidus(Pio))
        res = minimize_scalar(diff_function, bounds=(Pcm, Pc), method='bounded')
        return res.x

    def calculate_inner_core_radius(self):
        Pc  = self.stevenson['Pc'] 
        Rc  = self.outer_radius
        rho = self.params['rho']
        g   = self.stevenson['g']
        Pio = calculate_pressure_io_boundary()
        Ri  = np.sqrt(2*(Pc -Pio)*Rc/(rho*g))
        return self.set_inner_core_radius(Ri)

    def core_energy_balance(core_flux, delta_core_radius, p):
            return -core_flux*p['A']/(p['rho']*p['c']*p['V']*p['epsilon']-p['L+Eg']*p['rho']*p['A']*delta_core_radius)

T0 = [ 2000., 1000.]        
core = CoreLayer(0.0,2020.0e3)
mantle = ConvectiveLayer(2020.e3, 2440.0e3)

mercury = Planet( [core,mantle], T0)
       
# mercury.draw()

    
