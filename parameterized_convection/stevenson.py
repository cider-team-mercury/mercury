import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative

import planetary_energetics


class StevensonCoreLayer(planetary_energetics.Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
        planetary_energetics.Layer.__init__(self,inner_radius,outer_radius,params)
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
    
    def light_alloy_concentration(self):
        '''
        Equation (7) from Stevenson 1983
        '''
        x0 = self.stevenson['x0']
        Rc = self.outer_radius
        Ri = self.inner_radius
        return x0*(Rc**3)/(Rc**3-Ri**3)

    def stevenson_liquidus(self, Pio):
        '''
        Equation (3) from Stevenson 1983
        
        Calculates the liquidus temp for a given pressure at the inner core
        outer core boundary Pio
        '''
        x  = self.light_alloy_concentration()
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

    def energy_balance(self, T_cmb, core_flux):
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


class StevensonMantleLayer(planetary_energetics.Layer):
    def __init__(self,inner_radius,outer_radius, params={}):
        planetary_energetics.Layer.__init__(self,inner_radius,outer_radius,params)
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
        return p['Q0']*np.exp(-p['lambda']*time)*0.

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
        upper_boundary_delta_T = T_upper_mantle - self.surface_temperature
        lower_boundary_delta_T = T_cmb - T_lower_mantle
        assert( upper_boundary_delta_T > 0.0)
        assert( lower_boundary_delta_T > 0.0)
        delta_T_effective = upper_boundary_delta_T + lower_boundary_delta_T
        return p['g']*p['alpha']*( delta_T_effective)*np.power(self.thickness,3.)/(nu*p['K_diff'])
    
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
        delta_T_lower_boundary_layer = T_cmb - T_lower_mantle
        assert( delta_T_lower_boundary_layer > 0.0 )
        delta = np.power( p['Ra_boundary_crit']*nu_crit*p['K_diff']/(p['g']*p['alpha']*(delta_T_lower_boundary_layer)), 0.333 )
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

    def energy_balance(self, time, T_upper_mantle, T_cmb):
        p = self.stevenson
        mantle_surface_area = self.outer_surface_area
        core_surface_area   = self.inner_surface_area

        effective_heat_capacity = p['rho']*p['c']*p['mu']*self.volume
        internal_heat_energy = self.heat_production(time)*self.volume
        cmb_flux = self.lower_boundary_flux(T_upper_mantle, T_cmb)
        surface_flux = self.upper_boundary_flux(T_upper_mantle, T_cmb) 
        flux_energy = mantle_surface_area*surface_flux - core_surface_area*cmb_flux
        dTdt = (internal_heat_energy - flux_energy)/effective_heat_capacity
        return dTdt


mercury = planetary_energetics.Planet( [ StevensonCoreLayer( 0.0, 2020.0e3) , StevensonMantleLayer( 2020.e3, 2440.e3 ) ] )
t, y = mercury.integrate()
mercury.draw()
plt.plot( t, y[:,0])
plt.plot( t, y[:,1])
plt.show()

