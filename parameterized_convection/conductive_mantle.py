# - Mantle Energry Balance Model for paramterized convection. This mantle layer follows 
# those implemented in the Thermal Evolution studies of Morschhauser et al 2011 and
# Grott et al 2011.
#
# - Currently implements temperature dependent viscosity mantle convection in the 
# Stagnant Lid regime, in addition to accounting for crustal growth, and partial melting.
#
# - Citations:
#
# Morschhauser, A., Grott, M., & Breuer, D. (2011). Crustal recycling, mantle dehydration,
# and the thermal evolution of Mars. Icarus, 212(2), 541-558.
#
# Grott, M., Breuer, D., & Laneuville, M. (2011). Thermo-chemical evolution and global
# contraction of Mercury. Earth and Planetary Science Letters, 307(1), 135-146.
#
# Grasset, O., & Parmentier, E. M. (1998). Thermal convection in a volumetrically heated,
# infinite Prandtl number fluid with strongly temperature-dependent viscosity: Implications
# for planetary thermal evolution. Journal of Geophysical Research: Solid Earth (1978-2012),
# 103(B8), 18171-18181.
#
# Deschamps, Frederic, and Christophe Sotin. "Thermal convection in the outer shell  large
# icy satellites." Journal of Geophysical Research: Planets 991-2012) 106.E3 (2001): 5107-5121.
#
# Richter, Frank M. "Mantle convection models." Ann Rev of Earth and Planetary Sciences (1978).

import numpy as np
import scipy.integrate as integrate
from planetary_energetics import Layer
from mercury_parameters import mantle_params

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join('.', os.pardir)))
from mercury_interior_structure_model import model1

import mercury_mantle_melt_model as melt_model
import matplotlib.pyplot as plt
from heat_production import WD94 as heat_production_model
from scipy.constants import Julian_year

class ConductiveLayer(Layer):

    def __init__(self, inner_radius, outer_radius, params=mantle_params):

        Layer.__init__(self, inner_radius, outer_radius)

        self.params = { 'T_surface': 400.,
                        'mantle_density': 3400., 
                        'crustal_density': 2800.,
                        'core_density': 7200.,
                        'core_heat_capacity': 465.,
                        'mantle_thermal_conductivity': 4.0,
                        'crustal_thermal_conductivity': 4.0 }
        self.crustal_thickness = 50.e3
        self.radius_crust = self.outer_radius-self.crustal_thickness

        self.crustal_heating_rate = 1.e-7
        self.mantle_heating_rate = 1.e-8

        self.kc = self.params['crustal_thermal_conductivity']
        self.km = self.params['mantle_thermal_conductivity']
        self.qc = self.crustal_heating_rate
        self.qm = self.mantle_heating_rate

        self.core_energetic_model = model1
       

    def _compute_two_layer_diffusion_constants( self, T_cmb, T_surface):
    

        dT = T_cmb-T_surface

        delta_q = self.qc-self.qm
        k = self.kc/self.km

        alpha = np.power(self.radius_crust, 3.)*delta_q/(3*self.km)
        beta = np.power(self.radius_crust, 2)*(self.qm/self.km -self.qc/self.kc)/6 -alpha/self.radius_crust
        gamma = -(self.qm*np.power(self.inner_radius, 2.))/(6*self.km) + alpha/self.inner_radius + beta

        top = -self.qc*np.power(self.outer_radius, 2)/(6*self.kc)-gamma +dT
        bottom = k/self.inner_radius +(1-k)/self.radius_crust -1./self.outer_radius

        c1 = top/bottom
        c2 = T_cmb -gamma - (k/self.inner_radius +(1-k)/self.radius_crust)*c1
        m1 = alpha + k*c1
        m2 = beta + (1-k)*c1/self.radius_crust +c2

        return c1, c2, m1, m2

    def temperature_profile( self, radius, T_cmb, T_surface ):
        """
        :param radius:
        :param T_cmb:
        :param T_surface:
        :return:
        """
        c1, c2, m1, m2 = self._compute_two_layer_diffusion_constants( T_cmb, T_surface)

        crust_solution  = lambda r: (-self.qc*r*r/(6.*self.kc) + c1/r + c2) if((self.outer_radius  >= r)and(r >= self.radius_crust))  else 0.
        mantle_solution = lambda r: (-self.qm*r*r/(6.*self.km) + m1/r + m2) if((self.radius_crust >= r)and(r >= self.inner_radius))  else 0.
        return crust_solution(radius) + mantle_solution(radius)
 
    def heat_flux_profile( self, radius, T_cmb, T_surface ):

        c1, c2, m1, m2 = self._compute_two_layer_diffusion_constants( T_cmb, T_surface)
        temperature_gradient_crust = lambda r : -self.qc*r/(3*self.kc) - c1/(r*r) if r >= self.radius_crust and r <= self.outer_radius else 0.
        temperature_gradient_mantle = lambda r : -self.qm*r/(3*self.km) - m1/(r*r) if r >= self.inner_radius and r < self.radius_crust else 0.
       
        return -self.kc*temperature_gradient_crust(radius) - self.km*temperature_gradient_mantle(radius)

    def surface_heat_flux( self, T_cmb, T_surface ):
        return self.heat_flux_profile( self.radius, T_cmb, T_surface)
 
    def cmb_heat_flux( self, T_cmb, T_surface ):
        return self.heat_flux_profile( self.inner_radius, T_cmb, T_surface)

    def core_energy_balance(self, T_cmb, T_surface):
        thermal_energy_change, gravitational_energy_release, latent_heat, total_effective_heat_capacity,\
                radius_inner_core = self.core_energetic_model.get_effective_core_heat_capacity()
        lhs = self.params['core_density']*self.params['core_heat_capacity']\
                *(4./3.*np.pi*np.power(self.inner_radius, 3.)) -\
                gravitational_energy_release(T_cmb) - latent_heat(T_cmb)

        print "Radius: ", radius_inner_core(T_cmb)

        lower_heat_flux = self.cmb_heat_flux(T_cmb, T_surface)
        dTc_dt = -lower_heat_flux*self.inner_surface_area/lhs
        assert(dTc_dt<0)
        return -dTc_dt

    def energy_balance(self, T_cmb):
        dTc_dt = self.core_energy_balance(T_cmb, self.params['T_surface'] )
        return dTc_dt

    def integrate(self):
        def ODE(y, t):
           print "T_cmb     : ", y
           return self.energy_balance(y)

        times = np.linspace(0., Julian_year * 4.5e9, 1000)
        solution = integrate.odeint( ODE, 1600., times)
        return times, solution

     
radius_planet = 2440.e3
radius_cmb = 2020.e3
T_surface = 400.
T_cmb = 3000.
mercury_mantle = ConductiveLayer(radius_cmb, radius_planet)

times, solution = mercury_mantle.integrate()
plt.plot(times, solution)
plt.show()
