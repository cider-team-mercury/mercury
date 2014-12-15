import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative

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


    def set_boundary_temperatures(self,outer_temperature,inner_temperature): 
        '''
        All layers should be able to track the temperatures of the their outer and inner
        boundary.
        '''
        self.outer_temperature = outer_temperature
        self.inner_temperature = inner_temperature

    def ODE(y, t):
        raise NotImplementedError("Need to define an ODE")

    def lower_heat_flux_attempt (self):
        raise NotImplementedError("Need to define a heat flux function")

    def upper_heat_flux_attempt (self):
        raise NotImplementedError("Need to define a heat flux function")
   

class Planet(object):

    def __init__( self, layers):
        self.layers = layers
        self.Nlayers = len(layers)

        self.radius = self.layers[-1].outer_radius 
        self.volume = 4./3. * np.pi * self.radius**3

        self.core_layer = layers[0]
        self.mantle_layer = layers[1]

    def integrate(self, T_cmb_initial, T_mantle_initial, times):
        
        def ODE( temperatures, t ):
            dTmantle_dt = self.mantle_layer.energy_balance( t, temperatures[1], temperatures[0] )
            cmb_flux = -self.mantle_layer.lower_boundary_flux( temperatures[1], temperatures[0] )
            dTcore_dt = self.core_layer.energy_balance(temperatures[0], cmb_flux )
            return np.array([dTcore_dt, dTmantle_dt])

        solution = integrate.odeint( ODE, np.array([T_cmb_initial, T_mantle_initial]), times)
        return times, solution
        
       
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

