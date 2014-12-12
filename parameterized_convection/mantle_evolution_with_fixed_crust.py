# - Mantle Energry Balance Model for paramterized convection. This mantle layer follows 
# those implemented in the Thermal Evolution studies of Morschhauser et al 2011 and
# Grott et al 2011.
#
# - Currently implements temperature dependent viscosity mantle convection in the 
# Stagnant Lid regime with a fixed crustal thickness
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
# import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import scipy.optimize as opt
from scipy.misc import derivative
from planetary_energetics import Layer
from mercury_parameters import mantle_params, rho_core, core_heat_capacity, Radius

import mercury_mantle_melt_model as melt_model
import matplotlib.pyplot as plt
from heat_production import schubert_spoon_heating_model
from scipy.constants import Julian_year

class MantleLayer(Layer):
    """
    Mantle layer which includes the crustal evolution as in Morschhauser et al 2011
    
    The independent variable controlling the evolution in the mantle is the Temperature
    of the upper mantle(T_upper_mantle), which is the temperature at the top of the 
    convecting mantle, or equivently the temperature at the base of the upper mantle
    boundary layer.

    Relevant temperatures in order of increasing depth include:
    - The surface temperature is a fixed boundary condition that must be provided to 
      the mantle layer.

    - The temperature at the base of the stagnant lid, or equivently the temp. at the
      top of the upper mantle boundary layer.

    - The temperature at the base of the upper mantle layer, or equivalently the temp.
      at the top of the convecting mantle. This temperature is the dependent variable.

    - The temperature at the base of the convecting mantle, or equivently the temp.
      at the top of the lower mantle boundary layer.

    - The temperature at the core-mantle boundary, or equivaently the base of the 
      lower mantle boundary layer. This temperature is a boundary condition, and
      is provided to the mantle from the core. 
    """

    def __init__(self, inner_radius, outer_radius, crustal_thickness, params=mantle_params,
                 T_cmb_initial=None, T_mantle_initial=None, D_lid_initial=None):
        Layer.__init__(self, inner_radius, outer_radius, params)
        """
        Default Values for Mercury are loaded from the file mercury_paramaters, which are
        the same as in Grott et al 2011 which implements the model of Morchhauser et al 2011
        for Mercury rather than Mars.
        """
        self.params = params
        self.crustal_thickness = crustal_thickness
        #TODO Get cmb gravity from Sean's Model
        self.gravity_cmb = params['surface_gravity']
        self.initial_conditions = np.array([T_mantle_initial, T_cmb_initial, D_lid_initial])

    def effective_heat_capacity(self):
        """
        All terms multiplying the change in temperature with time(dT_dt), in equation (1) of Morschhauser et al 2011.
        
        For now we have ignored the effect of mantle melting, to include the effect of melting
        simply multiply by (1+St), where St is the Stefan number. See the LHS of equation (1)
        in Morschhauser et al 2011. The definition of the Stefan number is given in equation (2).
        """
        return self.params['density'] * self.params['heat_capacity'] * self.params['epsilon'] * self.volume

    def calculate_viscosity(self, temperature):
        """
        Temperature Dependent Viscosity, equation (7) in Morschhauser et al 2011. 

        We use the Arrhenius relationship to relate the viscosity at some reference temperature
        (T_ref) to the current upper mantle temperature. The reference viscosity for Mercury is
        unknown and varied by 3 orders of magnitude in Grott et al. 2011, from 10^19 to 10^22 Pa-s.
        :param temperature:
        """
        temperature_ref = self.params['reference_temperature']
        normalized_temp = (temperature_ref - temperature) / (temperature_ref * temperature)
        return self.params['reference_viscosity'] * np.exp(self.params['activation_energy'] *
                                                           normalized_temp / self.params['gas_constant'])

    def calculate_rayleigh_number(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        The Mantle Rayleigh Number, equation (6) in Morschhauser et al 2011.  
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        """
        coef = self.params['thermal_expansivity'] * self.params['density'] * self.params['surface_gravity']
        mu = self.calculate_viscosity(T_upper_mantle)
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)

        delta_temp = (T_upper_mantle - temperature_base_stagnant_lid) + (T_cmb - temperature_base_mantle)
  
        Ra =  coef * delta_temp * np.power(self.thickness - stagnant_lid_thickness, 3.) / (
              self.params['thermal_diffusivity'] * mu)
        assert( Ra > 0. ), "{},{},{},{}".format(T_upper_mantle, temperature_base_stagnant_lid, T_cmb, temperature_base_mantle)
        return Ra

    def calculate_critical_internal_rayleigh_number(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Critical Internal Rayleigh Number, equations (14,15) Morschhauser et al 2011.

        It was found that the local critical Rayleigh Number, for the lower boundary layer
        should be calculated using the Critical Internal Rayleigh Number, see Deschamps and 
        Sotin 2001.

        This empirical relation between the Internal Rayleigh Number and the Critical Rayleigh Number,
        is a power-law relation involving two fitting parameters:
                    Ra_i,crit = 0.28*(Ra_i)^0.21
        """
        hard_coded_coef = 0.28
        hard_coded_exponent = 0.21
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        delta_temp_internal = T_upper_mantle - self.params['surface_temperature'] + T_cmb - temperature_base_mantle
        viscosity = self.calculate_viscosity(T_upper_mantle)
        coef = self.params['thermal_expansivity'] * self.params['surface_gravity'] * self.params['density']
        Ra_internal = coef * delta_temp_internal * np.power(self.thickness, 3.) / (
            self.params['thermal_diffusivity'] * viscosity)
        return hard_coded_coef * np.power(Ra_internal, hard_coded_exponent)

    def calculate_temperature_base_stagnant_lid(self, T_upper_mantle):
        """
        Temperature at the base of the Stagnant Lid, equation (8) in Morschhauser et al 2011. 
        
        In previous numerical models and experiments, this temperature was found to be the temp.
        in which the viscosity had grown by an order of magnitude with respect to the convecting mantle.
        See Grasset and Parmentier 1998.
        """
        theta = self.params['empirical_spherical_stagnant_lid_param']
        temperature_drop = theta * self.params['gas_constant'] * np.power(T_upper_mantle, 2.) / \
                           self.params['activation_energy']
        return T_upper_mantle - temperature_drop

    def calculate_temperature_base_mantle(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Temperature at the base of the Mantle, equation (9) in Morschhauser et al 2011. 

        The temperature at the base of the mantle is found by determining the adiabatic increase in the mantle.
        However, to do this correctly we need to know the thickness of the Stagnant Lid, and the thickness of
        the boundary layers at the base of the mantle and the boundary layer at the base of the Stagnant Lid/top
        of the convecting mantle.
        """
        #TODO Should we iterate over upper_boundary_layer_thickness, etc?
#        upper_boundary_layer_thickness = self.calculate_upper_boundary_layer_thickness(T_upper_mantle, T_cmb,
#                                                                                       stagnant_lid_thickness)
#        lower_boundary_layer_thickness = self.calculate_lower_boundary_layer_thickness(T_upper_mantle, T_cmb,
#                                                                                       stagnant_lid_thickness)
        thickness_convecting_mantle = self.thickness - stagnant_lid_thickness# - upper_boundary_layer_thickness \
 #                                     - lower_boundary_layer_thickness
        g = self.params['surface_gravity']  # Is the Surface gravity the correct gravity here
        adiabatic_temperature_increase = self.params['thermal_expansivity'] * g * thickness_convecting_mantle * \
                                         T_upper_mantle / self.params['heat_capacity']
        return T_upper_mantle + adiabatic_temperature_increase

    def calculate_upper_boundary_layer_thickness(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Thickness of the boundary layer at the top of the convecting mantle, equation (12) in
        Morschhauser et al 2011. 

        The thickness of this boundary is calculated using the Nusselt-Rayleigh relation and
        critical boundary layer theory, see Turcotte and Schubert 2002.
        """
        beta = 1. / 3.
        Ra = self.calculate_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        return (self.thickness - stagnant_lid_thickness) * np.power(self.params['critical_rayleigh_number'] / Ra, beta)

    def calculate_lower_boundary_layer_thickness(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Thickness of the boundary layer at the base of the convecting mantle, equation (13) in
        Morschhauser et al 2011.

        Due to the temperature dependent viscosity this boundary layer is calculated using a
        "Local Critical Rayleigh Number" and the viscosity at the average temperature in the 
        boundary layer, see Richter et al 1978. 
        """
        Ra_i_crit = self.calculate_critical_internal_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        average_boundary_temperature = (temperature_base_mantle + T_cmb) / 2.0
        mu_crit = self.calculate_viscosity(average_boundary_temperature)
        delta_T = T_cmb - temperature_base_mantle
        numerator = self.params['thermal_diffusivity'] * self.params[
            'factor_pressure_dependent_viscosity'] * mu_crit * Ra_i_crit
        denominator = self.params['thermal_expansivity'] * self.params['density'] * self.gravity_cmb * delta_T
        return np.power(numerator / denominator, 1. / 3.)

    def calculate_upper_heat_flux(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Heat flow from the convecting mantle into the stagnant lid, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        upper_boundary_layer_thickness = self.calculate_upper_boundary_layer_thickness(T_upper_mantle, T_cmb,
                                                                                       stagnant_lid_thickness)
        return self.params['thermal_conductivity'] * (
            T_upper_mantle - temperature_base_stagnant_lid) / upper_boundary_layer_thickness

    def calculate_lower_heat_flux(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Heat flow out of the mantle, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        """
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        lower_boundary_layer_thickness = self.calculate_lower_boundary_layer_thickness(T_upper_mantle, T_cmb,
                                                                                       stagnant_lid_thickness)
        return self.params['thermal_conductivity'] * (T_cmb - temperature_base_mantle) / lower_boundary_layer_thickness

    def convert_radius_to_hydrostatic_pressure(self, radius):
        """
        Convert radius to hydrostatic pressure using P = rho*g*z
        :param radius:
        :return:
        """
        return (self.outer_radius-radius)*self.params['density']*self.params['surface_gravity']

    def stagnant_lid_geometry(self, stagnant_lid_thickness):
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        volume_stagnant_lid = 4./3.*np.pi*( np.power(self.outer_radius, 3.) - np.power(radius_stagnant_lid, 3.) )
        return radius_stagnant_lid, volume_stagnant_lid

    def crustal_geometry(self):
        radius_crust = self.outer_radius - crustal_thickness
        volume_crust = 4./3.*np.pi*( np.power(self.outer_radius, 3.) - np.power(radius_crust, 3.) )
        return radius_crust, volume_crust

    def get_stagnant_lid_thermal_profile(self, T_upper_mantle, stagnant_lid_thickness, volumetric_heating):
        """
        Get the radial conductive temperature profile of the stagnant lid, equation (5) Morschhauser et al (2011).

        Will replace with the two layer eventually...
        :param T_upper_mantle:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        #TODO Need to get analytical solution for two layer system
        k = self.params['thermal_conductivity']
        temp_surface = self.params['surface_temperature']
        radius_surface = self.outer_radius
        temp_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        stagnant_lid_radius, stagnant_lid_volume = self.stagnant_lid_geometry(stagnant_lid_thickness)
        crust_radius, crust_volume = self.crustal_geometry()
        Q = volumetric_heating*crust_volume
        coef1 = ( temp_surface - temp_base_stagnant_lid + Q/(6.*k)*( np.power(radius_surface, 2.) -
                                                                     np.power(stagnant_lid_radius, 2.))
                )/(1./radius_surface - 1./stagnant_lid_radius)
        coef2 = temp_surface - coef1/radius_surface + Q*np.power(radius_surface , 2.)/(6.*k)

        temperature_profile_as_function_of_radius = lambda r: -Q/(6*k)*r*r + coef1/r + coef2
        #r = np.linspace(self.outer_radius-stagnant_lid_thickness, self.outer_radius, 1000)
        #t = temperature_profile_as_function_of_radius(r)
        #plt.plot(t, r)
        #plt.show()
        return temperature_profile_as_function_of_radius

    def calculate_thermal_gradient_base_stagnat_lid(self, T_upper_mantle, stagnant_lid_thickness, volumetric_heating):
        """
        Calculate the thermal gradient at the base of the stagnant lid by solving the radial heat conduction
        equation in the stagnant lid, equation (5), and evaluating its derivative at the base of the stagnant.
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        temp_profile =self.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, volumetric_heating)
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        return derivative(temp_profile , radius_stagnant_lid, dx=1e-1 )



    def get_rate_of_stagnant_lid_growth(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, volumetric_heating):
        """
        The growth of the stagnant lid is determined by the energy balance at the base of the stagnant lid,
        equation (4) Morschhauser et al (2011).

        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        delta_T = T_upper_mantle - temperature_base_stagnant_lid
        lhs_coef = self.params['density']*self.params['heat_capacity']*delta_T
        flux_thermal_gradient = self.params['thermal_conductivity']*self.calculate_thermal_gradient_base_stagnat_lid(
            T_upper_mantle, stagnant_lid_thickness, volumetric_heating)
        upper_heat_flux = self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        print upper_heat_flux, flux_thermal_gradient
        dDlid_dt = (-upper_heat_flux - flux_thermal_gradient)/lhs_coef
        return dDlid_dt

    def energy_conservation_mantle(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, volumetric_heating):
        """
        Equation (1) in Morschhauser et al (2001),the energy conservation equation in the mantle to
        be solved to determine the thermal evolution of the planet.
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        surface_area_base_stagnant_lid = 4.*np.pi*radius_stagnant_lid
        volume_stagnant_lid = 4./3.*np.pi*(np.power(radius_stagnant_lid,3.) - np.power(self.inner_radius,3.))
        lhs_coef = self.params['density']*self.params['heat_capacity']*self.volume
        upper_heat_flux = self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        lower_heat_flux = self.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        crust_radius, crust_volume = self.crustal_geometry()
        dTm_dt = (upper_heat_flux*surface_area_base_stagnant_lid + lower_heat_flux*self.inner_surface_area +
                    volumetric_heating*crust_volume)/lhs_coef
        return dTm_dt

    def calculate_volumetric_heating(self, time):
        return schubert_spoon_heating_model(time)

    def core_energy_balance(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        lhs = rho_core*core_heat_capacity*4./3.*np.pi*np.power(self.inner_radius, 3.)
        lower_heat_flux = self.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        dTc_dt = -lower_heat_flux*self.inner_surface_area/lhs
        return dTc_dt

    def energy_balance(self, time, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        volumetric_heating = 0.0#self.calculate_volumetric_heating(time)
        dTm_dt = self.energy_conservation_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness, volumetric_heating)
        dDlid_dt = self.get_rate_of_stagnant_lid_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness, volumetric_heating)
        dTc_dt = self.core_energy_balance(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        #print np.array([dTm_dt, dTc_dt, dDlid_dt])
        return np.array([dTm_dt, dTc_dt, dDlid_dt])

    def integrate(self):
        def ODE(y, t):
            return self.energy_balance(t, y[0], y[1], y[2])

        times = np.linspace(0., Julian_year * 4.5e9, 10000)
        solution = integrate.odeint( ODE, self.initial_conditions, times)
        return times, solution

radius_planet = 2440.e3
radius_cmb = 2020.e3
crustal_thickness = 120.e3
initial_Tm = 2000.
initial_Tcmb = 2300.
initial_Dlid= 120.e3

merc = MantleLayer(radius_cmb, radius_planet, crustal_thickness, mantle_params, initial_Tcmb, initial_Tm, initial_Dlid)
#print merc.initial_conditions
#print merc.energy_balance(1.e8, initial_Tm, initial_Tcmb, initial_Dlid)
times, solution = merc.integrate()
#print times, solution
plt.figure()
plt.plot(times,solution[:, 0])
plt.plot(times,solution[:, 1])
#plt.plot(times,solution[:, 2])
plt.show()
