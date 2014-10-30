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
                'rho' : 3500.,
                'c'   : 1142.,
                'mu'  : 1.,
                'Q0'  : 1.7e-7, # - [W]/[m]
                'lambda' : 1.38e-17, # - [s]
                'A'      : 5.2e4, # - [k]
                'v0' : 4.0e3, # [m]^2/[s]
                'k'  : 4.0, # - [W]/[m]/[K]
                'beta' : 0.3, # - Ra exponent
                'alpha' : 2 * 10e-5, # - 1/[K]
                'g'     : 3.8, # - [m]/[s]/[s] 
                'K_diff' : 10.e-6, # - [m][m]/[s]
                'Ra_crit' : 500.,
                'Ra_boundary_crit' : 2000.,
                'T_surf' : 1073.
            }
        self.surface_temperature = self.stevenson['T_surf']

    ### We could code the integrals here. 
    def average_mantle_temp(self, T_upper_mantle):
        p = self.stevenson
        return  T_upper_mantle * p['mu']

    def kinematic_viscosity(self, T_upper_mantle):
        p = self.stevenson
        return p['v0']*np.exp(p['A']/T_upper_mantle)
    
    def heat_production(self, time):
        '''
        Equation (2) from Stevenson et al 1983
        '''
        p = self.stevenson
        return p['Q0']*np.exp(-p['lambda']*time)

    # - The Thickness used here is slightly wrong since we ignore the boundary layer thickness
    #   and extend to the CMB rather than the top of the boundary Layer since we don't know what
    #   it is yet. Not sure what Stevenson did originally, but I will iterate until the thickness of the
    #   lower layer and the temp are consistent eventually, though I don't think the drop in the adiabat is
    #   much for Mercury across the layer thickness
    def lower_mantle_temperature(self, T_upper_mantle):
        '''
        Adiabatic Temperature Increase from the temperature at the base of upper mantle boundary layer to
        the top of the lower boundary layer assuming negligable boundary layer thickness.
        '''
        p =self.stevenson
        return T_upper_mantle*( 1.0 + p['alpha']*p['g']*self.thickness/p['c'])
    
    def mantle_rayleigh_number(self, T_upper_mantle, T_cmb):
        '''
        Equation (19) Stevesnon et al 1983
        '''
        p = self.stevenson
        nu = self.kinematic_viscosity(T_upper_mantle)
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        delT_eff = (self.surface_temperature-T_upper_mantle)+(T_lower_mantle-T_cmb)
        return p['g']*p['alpha']*( delT_eff)*np.power(self.thickness,3)/(nu*p['K_diff'])
    
    def boundary_layer_thickness(self, Ra_mantle):
        '''
        Equation (18) Stevenson et al 1983
        '''
        p = self.stevenson
        return self.thickness*np.power(p['Ra_crit']/Ra_mantle,p['beta'])

    def upper_boundary_layer_thickness(self, T_upper_mantle, T_cmb):
        '''
        Use Equations (18,19) from Stevenson et al 1983 
        '''
        Ra = self.mantle_rayleigh_number(T_upper_mantle, T_cmb)
        return self.boundary_layer_thickness(Ra)
    
    def lower_boundary_layer_thickness(self, T_upper_mantle, T_cmb):
        '''
        Equations (20,21) Stevenson et al 1983
        '''
        p = self.stevenson
        T_lower_mantle = self.lower_mantle_temperature(T_upper_mantle)
        average_boundary_layer_temp = (T_upper_mantle + T_lower_mantle)/2
        nu_crit = self.kinematic_viscosity(average_boundary_layer_temp)
        delta = np.power( p['Ra_boundary_crit']*nu_crit*p['K_diff']/(p['g']*p['alpha']*(T_lower_mantle-T_cmb)), 0.333 )
        Ra_mantle = self.mantle_rayleigh_number(T_upper_mantle, T_cmb)
        return np.minimum(delta, self.boundary_layer_thickness(Ra_mantle) )

    def upper_boundary_flux(self, T_upper_mantle, T_cmb):
        thermal_conductivity = self.stevenson['k']
        delta_T = self.surface_temperature - T_upper_mantle
        upper_boundary_layer_thickness = self.upper_boundary_layer_thickness(T_upper_mantle, T_cmb)
        return -thermal_conductivity*delta_T/upper_boundary_layer_thickness

    def lower_boundary_flux(self, T_upper_mantle, T_cmb):
        thermal_conductivity = self.stevenson['k']
        delta_T = T_cmb - self.lower_mantle_temperature(T_upper_mantle)
        lower_boundary_layer_thickness = self.lower_boundary_layer_thickness(T_upper_mantle, T_cmb)
        return -thermal_conductivity*delta_T/lower_boundary_layer_thickness

    def mantle_energy_balance(self, time, T_upper_mantle, T_cmb):
        p = self.stevenson
        mantle_surface_area = self.outer_surface_area
        core_surface_area   = self.inner_surface_area

        effective_heat_capacity = p['rho']*p['c']*p['mu']*self.volume
        internal_heat_energy = self.heat_production(time)*self.volume
        cmb_flux = self.lower_boundary_flux(T_upper_mantle, T_cmb)
        surface_flux = self.upper_boundary_flux(T_upper_mantle, T_cmb) 
        flux_energy = mantle_surface_area*surface_flux - core_surface_area*cmb_flux
        print effective_heat_capacity, internal_heat_energy, flux_energy 
 
        dTdt = (internal_heat_energy - flux_energy)/effective_heat_capacity
        return dTdt

    def ODE( self, T_u_initial ):
        T_cmb = self.surface_temperature
        dTdt = lambda x, t : self.mantle_energy_balance( t, x, T_cmb )
        times = np.linspace( 0., 1.e9*np.pi*1.e7, 1000 )
        sol = integrate.odeint( dTdt, T_u_initial, times)
        y = sol
        return times, y

mantle = MantleLayer(2020.0e3, 2440.0e3)
t, y = mantle.ODE( 2000.0 )
plt.plot(t,y)
plt.show()
