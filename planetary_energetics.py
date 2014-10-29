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

    def __init__( self, layers, T0 ):
       self.layers = layers
       self.temperatures = T0
       self.Nlayers = len(layers)

       self.radius = self.layers[-1].outer_radius 
       self.volume = 4./3. * np.pi * self.radius**3

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


class CoreLayer(Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
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
            'alpha_c'   : 2.,
            'g'         : 3.8,
            'Tm0'       : 1880.0,
            'Tm1'       : 1.36/1.e12,
            'Tm2'       : -6.2/1.e12/1.e12,
            'Ta1'       : 8.0/1.e12,
            'Ta2'       : -3.9/1.e12/1.e12,
            'x0'        : 0.01,
            'Pcm'       : 10.0e9,
            'Pc'        : 40.0e9,
            'rho' : 7200.,
            'c'   : 465.,
            'L+Eg': 2.5e5,
            'mu' : 1.1
            }
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
        return  self.T_average / self.mu

    def stevenson_liquidus(self, Pio):
        '''
        Equation (3) from Stevenson 1983
        
        Calculates the liquidus temp for a given pressure at the inner core
        outer core boundary Pio
        '''
        x  = self.light_alloy
        p  = self.stevenson
        return p['Tm0']*(1.-p['alpha_c']*x)*(1. + p['Tm1']*Pio +p['Tm2']*Pio**2.)        
    
    def stevenson_adiabat(self,Pio, T_cmb):
        '''
        Equation (4) from Stevenson 1983

        Calculates adiabat temp for a given pressure at the inner core 
        outer core boundary Pio
        '''
        p = self.stevenson
        return T_cmb*(1.+p['Ta1']*Pio+p['Ta2']*Pio**2.)/(1.+p['Ta1']*p['Pcm']+p['Ta2']*p['Pcm']**2.)
    
    def calculate_pressure_io_boundary(self, T_cmb):
        p = self.stevenson
        opt_function = lambda Pio: (self.stevenson_adiabat(Pio, T_cmb)-self.stevenson_liquidus(Pio))
        if opt_function(p['Pc'])*opt_function(p['Pcm']) >= 0.:
            raise ValueError("OOGA BOOGA")
        else:
            res = opt.brentq(opt_function, p['Pc'], p['Pcm'])
            return res

    def inner_core_radius(self, T_cmb): 
        '''
        Equation 5 from Stevenson et al 1983
        '''
        p = self.stevenson
        Rc  = self.outer_radius
        Pio = self.calculate_pressure_io_boundary( T_cmb )
        Ri  = np.sqrt(2.*(p['Pc'] -Pio)*Rc/(p['rho']*p['g']))
        return Ri

    def core_energy_balance(self, core_flux, T_cmb):
        p = self.stevenson
        core_surface_area = self.outer_surface_area
          
        inner_core_surface_area = 0
        try:
            inner_core_surface_area = np.power(self.inner_core_radius(T_cmb), 2.0) * 4. * np.pi
        except ValueError:
            pass
  
        dRi_dTcmb = 0.
        try:
            dRi_dTcmb = derivative( self.inner_core_radius, T_cmb, dx=1.0)
        except ValueError:
            pass    
        thermal_energy_change = p['rho']*p['c']*self.volume*p['mu']
        latent_heat = -p['L+Eg'] * p['rho'] * inner_core_surface_area * dRi_dTcmb
        print thermal_energy_change, latent_heat, dRi_dTcmb
 
        
        dTdt = -core_flux * core_surface_area / (thermal_energy_change-latent_heat)
        return dTdt

    def ODE( self, T_cmb_initial ):
        cmb_flux = 2.e12/self.outer_surface_area
        dTdt = lambda x, t : self.core_energy_balance( cmb_flux, x )
        times = np.linspace( 0., 1.e9*np.pi*1.e7, 1000 )

        sol = integrate.odeint( dTdt, T_cmb_initial, times)
        y = sol
        return times, y

class MantleLayer(Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
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
                'rho' : 3500,
                'c'   : 1142,
                'mu'  : 1,
                'Q0'  : 1.7e7, # - [W]/[m]
                'lambda' : 1.38e-17 # - [s]
            }
        self.light_alloy = self.stevenson['x0']


    ### We could code the integrals here. 
    def upper_mantle_temp(self):
        return  self.T_average / self.mu

    def heat_production(self, time):
        p =self.stevenson
        return p['Q0']*np.exp(-p['lambda']*time)
    
    def mantle_energy_balance(self, surface_flux, cmb_flux, T_upper_mantle):
        p = self.stevenson
        mantle_surface_area = self.outer_surface_area
        core_surface_area   = self.inner_surface_area

        thermal_energy_change = p['rho']*p['c']*p['mu']*self.volume
        heating = 0*self.volume
        flux = mantle_surface_area*surface_flux - core_surface_area*cmb_flux
        latent_heat = -p['L+Eg'] * p['rho'] * inner_core_surface_area * dRi_dTcmb
        print thermal_energy_change, latent_heat, dRi_dTcmb
 
        
        dTdt = -core_flux * core_surface_area / (thermal_energy_change-latent_heat)
        return dTdt

    def ODE( self, T_cmb_initial ):
        cmb_flux = 2.e12/self.outer_surface_area
        dTdt = lambda x, t : self.core_energy_balance( cmb_flux, x )
        times = np.linspace( 0., 1.e9*np.pi*1.e7, 1000 )

        sol = integrate.odeint( dTdt, T_cmb_initial, times)
        y = sol
        return times, y



core = CoreLayer(0.0,2020.0e3)
t, y = core.ODE( 2000. )
plt.plot(t, y)
plt.show()

