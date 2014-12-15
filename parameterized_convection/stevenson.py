import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative
from mercury_parameters import rho_core, core_heat_capacity
import planetary_energetics
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('.', os.pardir)))
from mercury_interior_structure_model import model1
from scipy.constants import Julian_year
import prettyplotlib as ppl

class StevensonCoreLayer(planetary_energetics.Layer):
    def __init__(self, inner_radius, outer_radius, params={}):
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
        print (thermal_energy_change-latent_heat)
        dTdt = -core_flux * core_surface_area / (thermal_energy_change-latent_heat)
        print "dTc_dt :", dTdt
        print "Left Hand Side", (thermal_energy_change-latent_heat)
        return dTdt

class realisticCoreLayer(planetary_energetics.Layer):
    def __init__(self,inner_radius,outer_radius, core_energetic_model, params={}):
        planetary_energetics.Layer.__init__(self, inner_radius,outer_radius,params)
        self.core_energetic_model = core_energetic_model
        self.density = params['density']
        self.heat_capacity = params['heat_capacity']

    def energy_balance(self, T_cmb, core_flux):
        thermal_energy_change, gravitational_energy_release, latent_heat, total_effective_heat_capacity,\
            radius_inner_core = self.core_energetic_model.get_effective_core_heat_capacity()

        lhs = self.density*self.heat_capacity*(4./3.*np.pi*self.volume) -\
            gravitational_energy_release(T_cmb) - latent_heat(T_cmb)
        if(lhs<0):
            print "I am the lhs: ", lhs
            print  thermal_energy_change(T_cmb), gravitational_energy_release(T_cmb), latent_heat(T_cmb)
        dTc_dt = -100.*core_flux*self.outer_surface_area/lhs
        #print "dTc_dt", dTc_dt
        #print "Effective Heat Capacity: ", lhs
        assert(dTc_dt<0)
        return dTc_dt

    def inner_radius(self, T_upper_mantle):
        thermal_energy_change, gravitational_energy_release, latent_heat, total_effective_heat_capacity,\
            radius_inner_core = self.core_energetic_model.get_effective_core_heat_capacity()
        return radius_inner_core(T_upper_mantle)


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
        temperature_lower_mantle = T_upper_mantle*( 1.0 + p['alpha']*p['g']*self.thickness/p['c'])
  #      print "T_upper_mantle :", temperature_lower_mantle
        return temperature_lower_mantle
    
    def mantle_rayleigh_number(self, T_upper_mantle, T_cmb):
        '''
        Equation (19) Stevesnon et al 1983
        '''
        p = self.stevenson
        nu = self.kinematic_viscosity(T_upper_mantle)
        T_lower_mantle = T_upper_mantle#self.lower_mantle_temperature(T_upper_mantle)
        upper_boundary_delta_T = T_upper_mantle - self.surface_temperature
        lower_boundary_delta_T = T_cmb - T_lower_mantle
        assert( upper_boundary_delta_T > 0.0)
        assert( lower_boundary_delta_T > 0.0)
        delta_T_effective = upper_boundary_delta_T + lower_boundary_delta_T
        Ra = p['g']*p['alpha']*( delta_T_effective)*np.power(self.thickness,3.)/(nu*p['K_diff'])
 #       print "Rayleigh Number :", Ra
        assert(Ra>0.0)
        return Ra
    
    def boundary_layer_thickness(self, Ra_mantle):
        '''
        Equation (18) Stevenson et al 1983
        '''
        p = self.stevenson
        boundary_layer_thickness = self.thickness*np.power(p['Ra_crit']/Ra_mantle,p['beta'])
#        print "Boundary Layer Thickness :", boundary_layer_thickness
        assert(boundary_layer_thickness>0)
        return boundary_layer_thickness

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
        average_boundary_layer_temp = (T_cmb + T_lower_mantle)/2
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


#mercury = planetary_energetics.Planet( [ StevensonCoreLayer( 0.0, 2020.0e3) , StevensonMantleLayer( 2020.e3, 2440.e3 ) ] )
mercury = planetary_energetics.Planet( [ realisticCoreLayer( 0.0, 2020.0e3, model1, {'density': rho_core, 'heat_capacity': core_heat_capacity}) , StevensonMantleLayer( 2020.e3, 2440.e3 ) ] )
times = np.linspace( 0., Julian_year*np.pi*1.e7)

T_excess = 800.
times = np.linspace(0.0, 4.5*1.e9*Julian_year)
T_mantle_initial = 1800.


T_cmb_initial = T_mantle_initial + T_excess
Tm_low = T_mantle_initial - 100
Tcmb_low = T_cmb_initial - 100
Tm_high = T_mantle_initial + 100
Tcmb_high = T_cmb_initial + 100

t, y_low = mercury.integrate(Tcmb_low, Tm_low, times)
t, y_high = mercury.integrate(Tcmb_high, Tm_high, times)
t, y = mercury.integrate(T_cmb_initial, T_mantle_initial, times)
t = t/1.e9*Julian_year;
c = [ppl.colors.set2[0], ppl.colors.set2[1]]
plt.figure()
ppl.fill_between(t, y_low[:,0], y_high[:,0], color=c[0], label=r'CMB Temperature \pm 100 K')
plt.plot(t, y_low[:,0],'--k')
plt.plot(t, y_high[:,0],'--k')
ppl.fill_between(t, y_low[:,1], y_high[:,1], color=c[1], label=r'Upper Mantle Temperature \pm 100 K')
plt.plot(t, y_low[:,1],'--k')
plt.plot(t, y_high[:,1],'--k')
#mercury.draw()

ppl.plot( t, y[:,0], lw =3, color=c[0], label = 'CMB Temperature')
ppl.plot( t, y[:,0],  '--k',lw=1)
ppl.plot( t, y[:,1],  lw=3, color=c[1], label = 'Upper Mantle Temperature')
ppl.plot( t, y[:,1],  '--k',lw=1)
plt.xlabel('Time [Ga]')
plt.ylabel('Temperature [K]')
plt.title("Thermal Evolution Mercury")
ppl.legend(loc=1)
plt.savefig('thermal_evolution.png')

thermal_energy_change, gravitational_energy_release, latent_heat, total_effective_heat_capacity,\
            radius_inner_core = model1.get_effective_core_heat_capacity()
grav = []
total = []
latent = []
radius = []
for temp in y[:,1]:
    radius.append(radius_inner_core(temp)/1000.)
    grav.append(-gravitational_energy_release(temp)*1.e-26)
    #print "LATENT", latent_heat(temp)
    #print "Grav", gravitational_energy_release(temp)
    latent.append(-latent_heat(temp)*1.e-26)
    total.append(-total_effective_heat_capacity(temp)*1.e-26)

plt.figure()
ppl.plot(t,radius, label="Inner Core Radius")
plt.xlabel('Time [Ga]')
plt.ylabel('Inner Core Radius [kg]')
plt.title("Inner Core Growth")
ppl.legend()
plt.savefig('inner_core_growth.png')

plt.figure()
ppl.plot(t,grav,  label="Gravitational Energy Release")
ppl.plot(t,latent, label="Latent Heat Release")
ppl.plot(t,total, label="Total Effective Heat Capacity")
plt.xlabel('Time [Ga]')
plt.ylabel(r'\frac{dE}{dT_cmb} [W/K] * 10^{-26}')
plt.title(r'Core Energetics')
ppl.legend()
plt.savefig('core_energetics.png')



