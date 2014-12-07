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
# import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import scipy.optimize as opt
from scipy.misc import derivative
from planetary_energetics import Layer
from mercury_parameters import mantle_params
import mercury_mantle_melt_model as melt_model
import matplotlib.pyplot as plt
from heat_production import WD94 as heat_production_model

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

    def __init__(self, inner_radius, outer_radius, params=mantle_params):
        Layer.__init__(self, inner_radius, outer_radius, params)
        """
        Default Values for Mercury are loaded from the file mercury_paramaters, which are
        the same as in Grott et al 2011 which implements the model of Morchhauser et al 2011
        for Mercury rather than Mars.
        """
        self.params = params

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

    def calculate_rayleigh_number(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        The Mantle Rayleigh Number, equation (6) in Morschhauser et al 2011.  
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param gravity_cmb:
        """
        coef = self.params['thermal_expansivity'] * self.params['density'] * self.params['surface_gravity']
        mu = self.calculate_viscosity(T_upper_mantle)
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                                         gravity_cmb)
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)

        delta_temp = (T_upper_mantle - temperature_base_stagnant_lid) + (T_cmb - temperature_base_mantle)
  
        Ra =  coef * delta_temp * np.power(self.thickness - stagnant_lid_thickness, 3.) / (
              self.params['thermal_diffusivity'] * mu)
        assert( Ra > 0. )
        return Ra

    def calculate_critical_internal_rayleigh_number(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
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
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                                         gravity_cmb)
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

    def calculate_temperature_base_mantle(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        Temperature at the base of the Mantle, equation (9) in Morschhauser et al 2011. 

        The temperature at the base of the mantle is found by determining the adiabatic increase in the mantle.
        However, to do this correctly we need to know the thickness of the Stagnant Lid, and the thickness of
        the boundary layers at the base of the mantle and the boundary layer at the base of the Stagnant Lid/top
        of the convecting mantle.
        """
        #TODO Should we iterate over upper_boundary_layer_thickness, etc?
#        upper_boundary_layer_thickness = self.calculate_upper_boundary_layer_thickness(T_upper_mantle, T_cmb,
#                                                                                       stagnant_lid_thickness,
#                                                                                       gravity_cmb)
#        lower_boundary_layer_thickness = self.calculate_lower_boundary_layer_thickness(T_upper_mantle, T_cmb,
#                                                                                       stagnant_lid_thickness,
#                                                                                       gravity_cmb)
        thickness_convecting_mantle = self.thickness - stagnant_lid_thickness# - upper_boundary_layer_thickness \
 #                                     - lower_boundary_layer_thickness
        g = self.params['surface_gravity']  # Is the Surface gravity the correct gravity here
        adiabatic_temperature_increase = self.params['thermal_expansivity'] * g * thickness_convecting_mantle * \
                                         T_upper_mantle / self.params['heat_capacity']
        return T_upper_mantle + adiabatic_temperature_increase

    def calculate_upper_boundary_layer_thickness(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        Thickness of the boundary layer at the top of the convecting mantle, equation (12) in
        Morschhauser et al 2011. 

        The thickness of this boundary is calculated using the Nusselt-Rayleigh relation and
        critical boundary layer theory, see Turcotte and Schubert 2002.
        """
        beta = 1. / 3.
        Ra = self.calculate_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb)
        return (self.thickness - stagnant_lid_thickness) * np.power(self.params['critical_rayleigh_number'] / Ra, beta)

    def calculate_lower_boundary_layer_thickness(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        Thickness of the boundary layer at the base of the convecting mantle, equation (13) in
        Morschhauser et al 2011.

        Due to the temperature dependent viscosity this boundary layer is calculated using a
        "Local Critical Rayleigh Number" and the viscosity at the average temperature in the 
        boundary layer, see Richter et al 1978. 
        """
        Ra_i_crit = self.calculate_critical_internal_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                                     gravity_cmb)
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                                         gravity_cmb)
        average_boundary_temperature = (temperature_base_mantle + T_cmb) / 2.0
        mu_crit = self.calculate_viscosity(average_boundary_temperature)
        delta_T = T_cmb - temperature_base_mantle
        numerator = self.params['thermal_diffusivity'] * self.params[
            'factor_pressure_dependent_viscosity'] * mu_crit * Ra_i_crit
        denominator = self.params['thermal_expansivity'] * self.params['density'] * gravity_cmb * delta_T
        return np.power(numerator / denominator, 1. / 3.)

    def calculate_upper_heat_flux(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        Heat flow from the convecting mantle into the stagnant lid, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        upper_boundary_layer_thickness = self.calculate_upper_boundary_layer_thickness(T_upper_mantle, T_cmb,
                                                                                       stagnant_lid_thickness,
                                                                                       gravity_cmb)
        return self.params['thermal_conductivity'] * (
            T_upper_mantle - temperature_base_stagnant_lid) / upper_boundary_layer_thickness

    def calculate_lower_heat_flux(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb):
        """
        Heat flow out of the mantle, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        """
        temperature_base_mantle = self.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                                         gravity_cmb)
        lower_boundary_layer_thickness = self.calculate_lower_boundary_layer_thickness(T_upper_mantle, T_cmb,
                                                                                       stagnant_lid_thickness,
                                                                                       gravity_cmb)
        return self.params['thermal_conductivity'] * (T_cmb - temperature_base_mantle) / lower_boundary_layer_thickness

    def convert_radius_to_hydrostatic_pressure(self, radius):
        """
        Convert radius to hydrostatic pressure using P = rho*g*z
        :param radius:
        :return:
        """
        return (self.outer_radius-radius)*self.params['density']*self.params['surface_gravity']

    def get_stagnant_lid_thermal_profile(self, T_upper_mantle, stagnant_lid_thickness, mantle_heat_production):
        """
        Get the radial conductive temperature profile of the stagnant lid, equation (5) Morschhauser et al (2011).

        Will replace with the two layer eventually...
        :param T_upper_mantle:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        Q = mantle_heat_production
        k = self.params['thermal_conductivity']
        temp_surface = self.params['surface_temperature']
        radius_surface = self.outer_radius
        temp_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        radius_stagnant_lid = radius_surface - stagnant_lid_thickness
        coef1 = ( temp_surface - temp_base_stagnant_lid + Q/(6.*k)*( np.power(radius_surface, 2.) -
                                                                     np.power(radius_stagnant_lid, 2.))
                )/(1./radius_surface - 1./radius_stagnant_lid)
        coef2 = temp_surface - coef1/radius_surface + Q*np.power(radius_surface , 2.)/(6.*k)

        temperature_profile_as_function_of_radius = lambda r: -Q/(6*k)*r*r + coef1/r + coef2
        return temperature_profile_as_function_of_radius

    def calculate_thermal_gradient_base_stagnat_lid(self, T_upper_mantle, stagnant_lid_thickness, mantle_heat_production):
        """
        Calculate the thermal gradient at the base of the stagnant lid by solving the radial heat conduction
        equation in the stagnant lid, equation (5), and evaluating its derivative at the base of the stagnant.
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        temp_profile =self.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        return derivative(temp_profile , radius_stagnant_lid, dx=1e-1 )

    def get_mantle_solidus(self, crustal_thickness):
        """
        Calculate the solidus temperature of peridotite as a cubic function of radius,
        equation (16) in Morschhauser et al (2011). Also accounts for the increase in
        mantle solidus due to the depletion of crustal components, equations (18, 19).

        We assume the pressure in the stagnant lid is hydrostatic.

        :param pressure:
        :param crustal_thickness:
        :return:
        """

        solidus_as_function_of_radius = lambda r: \
            melt_model.mantle_solidus(self.convert_radius_to_hydrostatic_pressure(r), crustal_thickness,
                                      self.inner_radius, self.outer_radius)
        return solidus_as_function_of_radius

    def get_mantle_liquidus(self):
        """
        Calculate the solidus temperature of peridotite as a cubic function of radius,
        equation (17) in Morschhauser et al (2011).

        We assume the pressure in the stagnant is lid is hydrostatic.
        :return:
        """
        liquidus_as_function_of_radius = lambda r: \
            melt_model.peridotite_liquidus(self.convert_radius_to_hydrostatic_pressure(r))
        return liquidus_as_function_of_radius

    def calculate_volumetric_degree_melting(self, T_upper_mantle, stagnant_lid_thickness, mantle_heat_production):
        """
        The Volumeterically averaged degree of melting, equation (20) Morschhauser et al (2011).

        Partial Melting in the mantle, melt extraction, and crustal formation have a been
        shown to have a large effect on the thermal evolution of Mars and Mercury.
        These models follow the work of Breuer and Spohn (2003, 2006), but are modified to
        take into account the solidus increase due to mantle depletion. The presence of
        partial melt in the mantle depends on the solidus temperature of its constituents,
        here we assume the first melting mantle component is peridotite and utilize the
        parameterized solidus from Takahashi (1990).
        """
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        radius_planet_surface = self.outer_radius
        T = self.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        T_sol = self.get_mantle_solidus(crustal_thickness)
        T_liq = self.get_mantle_liquidus()

        do_we_have_melt = lambda r : ( 1.0 if T(r) > T_sol(r) else 0.0 )
        are_we_less_than_six_GPa = lambda r : (1.0 if self.convert_radius_to_hydrostatic_pressure(r) < 6.e9 else 0.0 ) 
 
        def melt_fraction_integrand(r):
            melt_fraction = (T(r) - T_sol(r))/( T_liq(r) - T_sol(r) )
            if ( T(r) > T_liq(r) ):
                melt_fraction = 1.0
            melt_fraction = melt_fraction*do_we_have_melt(r)*are_we_less_than_six_GPa(r)
            return melt_fraction

        def melt_volume_integrand(r):
            return 1.0 * do_we_have_melt(r)*are_we_less_than_six_GPa(r)

         
        fraction,_ = integrate.quad( lambda r : 4.0*np.pi*melt_fraction_integrand(r)*r*r, radius_stagnant_lid, radius_planet_surface)
        melt_volume,_ = integrate.quad( lambda r : 4.0*np.pi*melt_volume_integrand(r)*r*r, radius_stagnant_lid, radius_planet_surface)

        melt_degree = fraction/melt_volume
        return melt_degree, melt_volume

    def get_derivative_degree_melting(self, T_upper_mantle, stagnant_lid_thickness, mantle_heat_production):
        """
        Calcularte the rate of change of the degreee of melting with change in upper mantle temperature.

        This derivtive primarily used to calculate the Stefan number, equation (2) in Morschhauser et al (2001).
        :param T_upper_mantle:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        degree_melting_funcion_of_T = lambda T: self.calculate_volumetric_degree_melting(T, stagnant_lid_thickness,
                                                                                         mantle_heat_production)[0]
        return derivative(degree_melting_funcion_of_T, T_upper_mantle, dx=1.e-1)

    def calculate_stefan_number(self, T_upper_mantle, stagnant_lid_thickness, mantle_heat_production):
        """
        Calculate the Stefan Number from the average degree of melting, equation (2) Morschhauser et al (2011).
        :param T_upper_mantle:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        L = self.params['latent_heat_melting_crust']
        cm = self.params['heat_capacity']
        Rl = self.outer_radius-stagnant_lid_thickness
        Vl = 4./3.*np.pi*(np.power(Rl,3.)-np.power(self.inner_radius,3.))
        meltzone_volume = self.calculate_volumetric_degree_melting(T_upper_mantle, stagnant_lid_thickness,
                                                                   mantle_heat_production)[1]
        dma_dTm = self.get_derivative_degree_melting(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        St = (L/cm)*(meltzone_volume/Vl)*dma_dTm
        return St

    def get_rate_of_crustal_growth(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb,
                                   mantle_heat_production):
        """
        Calculate the rate of crustal growth, equation (21) Morschhauser et al (2011).

        We assume that the crust will be distributed uniformally on the planetary surface and that the timescale for
        melt extraction is limited by the rate at which undepleted mantle can be supplied to the meltzone, which
        occurs at the rate of mantle convection velocity scale.
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param gravity_cmb:
        :param mantle_heat_production:
        :return:
        """
        u0 = self.params['mantle_convection_speed_scale']
        Ra = self.calculate_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb)
        Ra_crit = self.params['critical_rayleigh_number']
        beta = 1. / 3.
        degree_melting, melt_volume = self.calculate_volumetric_degree_melting(T_upper_mantle, stagnant_lid_thickness,
                                                                               mantle_heat_production)
        mantle_convection_velocity = u0*np.power(Ra/Ra_crit, 2*beta)
        dDcrust_dt = mantle_convection_velocity*degree_melting*melt_volume/(4.*np.pi*np.power(self.outer_radius,3.))
        return dDcrust_dt

    def get_rate_of_stagnant_lid_growth(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb,
                                        mantle_heat_production):
        """
        The growth of the stagnant lid is determined by the energy balance at the base of the stagnant lid,
        equation (4) Morschhauser et al (2011).

        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param gravity_cmb:
        :param mantle_heat_production:
        :return:
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        delta_T = T_upper_mantle - temperature_base_stagnant_lid
        lhs_coef = self.params['density']*self.params['heat_capacity']*delta_T

        flux_thermal_gradient = self.params['heat_capacity']*self.calculate_thermal_gradient_base_stagnat_lid(
            T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        crustal_growth_rate = self.get_rate_of_crustal_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                               gravity_cmb, mantle_heat_production)
        volcanic_heat_piping = self.params['crustal_density']*crustal_growth_rate*\
                               (self.params['latent_heat_melting_crust'] + self.params['magma_heat_capacity']*delta_T)
        upper_heat_flux = self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb)

        dDlid_dt = (-upper_heat_flux + volcanic_heat_piping - flux_thermal_gradient)/lhs_coef
        return dDlid_dt

    def energy_conservation_mantle(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb,
                                        mantle_heat_production):
        """
        Equation (1) in Morschhauser et al (2001),the energy conservation equation in the mantle to
        be solved to determine the thermal evolution of the planet.
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param gravity_cmb:
        :param mantle_heat_production:
        :return:
        """
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        stefan_number = self.calculate_stefan_number(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        surface_area_base_stagnant_lid = 4.*np.pi*radius_stagnant_lid
        volume_stagnant_lid = 4./3.*np.pi*(np.power(radius_stagnant_lid,3.) - np.power(self.inner_radius,3.))
        print "Stefan Number is: "+str(stefan_number)
        lhs_coef = self.params['density']*self.params['heat_capacity']*(1. + stefan_number)
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)
        delta_T = T_upper_mantle - temperature_base_stagnant_lid

        crustal_growth_rate = self.get_rate_of_crustal_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                               gravity_cmb, mantle_heat_production)

        volcanic_heat_piping = self.params['crustal_density']*crustal_growth_rate*\
                               (self.params['latent_heat_melting_crust'] + self.params['magma_heat_capacity']*delta_T)
        upper_heat_flux = self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb)
        lower_heat_flux = self.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb)

        dDlid_dt = ((upper_heat_flux + volcanic_heat_piping)*surface_area_base_stagnant_lid +
                    lower_heat_flux*self.inner_surface_area + mantle_heat_production*volume_stagnant_lid)/lhs_coef
        return dDlid_dt

    def heat_production(self,T_upper_mantle, stagnant_lid_thickness):
        """
        The amount of radiogenic heating depends on the bulk concentration of radioactive
        nuclides and their distribution between the crust and mantle. The extraction of
        radioactive elements from the mantle is calculated self-consistently assuming
        accumulted fractional melting.
        :return:
        """
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        radius_planet_surface = self.outer_radius
        T = self.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
        T_sol = self.get_mantle_solidus(crustal_thickness)
        T_liq = self.get_mantle_liquidus()

        do_we_have_melt = lambda r : ( 1.0 if T(r) > T_sol(r) else 0.0 )
        are_we_less_than_six_GPa = lambda r : (1.0 if self.convert_radius_to_hydrostatic_pressure(r) < 6.e9 else 0.0 )

        def depth_dependent_melt_fraction(r):
            melt_fraction = (T(r) - T_sol(r))/( T_liq(r) - T_sol(r) )
            if ( T(r) > T_liq(r) ):
                melt_fraction = 1.0
            melt_fraction = melt_fraction*do_we_have_melt(r)*are_we_less_than_six_GPa(r)
            return melt_fraction

        F = lambda r : depth_dependent_melt_fraction(r)
        X_uranium =  lambda r: heat_production_model.uranium.liquid_partioning(depth_dependent_melt_fraction(r))
        X_potassium = lambda r: heat_production_model.potassium.liquid_partioning(depth_dependent_melt_fraction(r))
        X_thorium = lambda r: heat_production_model.thorium.liquid_partioning(depth_dependent_melt_fraction(r))
        fraction,_ = integrate.quad( , radius_stagnant_lid, radius_planet_surface)











radius_planet = 2440e3
stagnant_lid_thickness = 200e3
radius_stagnant_lid = radius_planet - stagnant_lid_thickness

radius_cmb = 2020e3
T_upper_mantle = 2000
T_cmb = 3000

crustal_thickness = 12e3
mantle_heat_production = 0.000001
mercury_mantle = MantleLayer(radius_cmb, radius_planet)
temp_func = mercury_mantle.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
liquidus_func = mercury_mantle.get_mantle_liquidus()
solidus_func  = mercury_mantle.get_mantle_solidus(crustal_thickness)
density = mercury_mantle.params['density']
gravity = mercury_mantle.params['surface_gravity']
gravity_cmb = mercury_mantle.params['surface_gravity']
r_solution = np.linspace(radius_stagnant_lid, radius_planet, 1000)
t = np.empty_like(r_solution)
t_peridotite_solidus = np.empty_like(r_solution)
t_solidus = np.empty_like(r_solution)
t_liquidus = np.empty_like(r_solution)
int = np.empty_like(r_solution)

for i, r in enumerate(r_solution):
    pressure = (radius_planet-r)*density*gravity
    t[i] = temp_func(r)
    t_solidus[i] = solidus_func(r)
    t_liquidus[i] = liquidus_func(r)
    t_peridotite_solidus[i] = melt_model.peridotite_solidus(pressure)
    int[i] = (t[i]-t_solidus[i])/(t_liquidus[i]-t_solidus[i])

plt.figure()
plt.plot(r_solution,t, label='Temperature Profile')
plt.plot(r_solution,t_solidus,label="Depleted Mantle Solidus")
plt.plot(r_solution,t_peridotite_solidus, label="Peridotite Solidus")
plt.plot(r_solution,t_liquidus, label="Peridotite Liquiduis")
plt.legend()

plt.figure()
plt.plot(r_solution, int)
plt.show()
dDcrust_dt = mercury_mantle.get_rate_of_crustal_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb, mantle_heat_production)
dDlid_dt = mercury_mantle.get_rate_of_stagnant_lid_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb, mantle_heat_production)
dTm_dt = mercury_mantle.energy_conservation_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness, gravity_cmb,mantle_heat_production)
degree, dv = mercury_mantle.calculate_volumetric_degree_melting(T_upper_mantle, stagnant_lid_thickness, mantle_heat_production)
