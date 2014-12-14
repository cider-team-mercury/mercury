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
#
# Hauck II, Steven A., et al. "Internal and tectonic evolution of Mercury."
# Earth and Planetary Science Letters 222.3 (2004): 713-728.

import numpy as np
# import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import scipy.optimize as opt
from scipy.misc import derivative
from planetary_energetics import Layer
from mercury_parameters import mantle_params, rho_core, core_heat_capacity, Radius

import mercury_mantle_melt_model as melt_model
import matplotlib.pyplot as plt
from heat_production import WD94, schubert_spoon_heating_model
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

    def __init__(self, inner_radius, outer_radius, crustal_thickness, params, radiogenic_heating_model,
                 T_cmb_initial=None, T_mantle_initial=None, D_lid_initial=None):
        Layer.__init__(self, inner_radius, outer_radius, params)
        """
        Default Values for Mercury are loaded from the file mercury_paramaters, which are
        the same as in Grott et al 2011 which implements the model of Morchhauser et al 2011
        for Mercury rather than Mars.
        """
        self.params = params
        self.crustal_thickness = crustal_thickness
        self.radiogenic_heating_model = radiogenic_heating_model
        #TODO Get cmb gravity from Sean's Model
        self.gravity_cmb = params['surface_gravity']
        self.initial_conditions = np.array([T_mantle_initial, T_cmb_initial, D_lid_initial])
        self.set_melt_fraction()
        self.time = 0.0

    def effective_heat_capacity(self, volume_convecting_mantle):
        """
        All terms multiplying the change in temperature with time(dT_dt), in equation (1) of Morschhauser et al 2011.
        
        For now we have ignored the effect of mantle melting, to include the effect of melting
        simply multiply by (1+St), where St is the Stefan number. See the LHS of equation (1)
        in Morschhauser et al 2011. The definition of the Stefan number is given in equation (2).
        """
        return self.params['density'] * self.params['heat_capacity'] * self.params['epsilon'] * volume_convecting_mantle

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
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle, stagnant_lid_thickness)

        delta_temp = (T_upper_mantle - temperature_base_stagnant_lid) + (T_cmb - temperature_base_mantle)

        radius_stagnant_lid, volume_stagnant_lid = self.stagnant_lid_geometry(stagnant_lid_thickness)
        assert(radius_stagnant_lid > self.inner_radius), "TIME STAGNANT LID GREW UP TO BE THE MANTLE: {}".format(self.time)
        Ra =  coef * delta_temp * np.power(radius_stagnant_lid - self.inner_radius, 3.) / (
              self.params['thermal_diffusivity'] * mu)
        assert( Ra > 0. ), "{},{},{},{}".format(T_upper_mantle, temperature_base_stagnant_lid, T_cmb, temperature_base_mantle)
        #print "The Rayleigh: ", Ra
        #print "The Viscosity: ", mu
        #assert Ra > 4.e3
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
        #print "Temp. Base Mantle: ", temperature_base_mantle
        #print "Temp. CMB: ", T_cmb
        delta_temp_internal = T_upper_mantle - self.params['surface_temperature'] + T_cmb - temperature_base_mantle
        viscosity = self.calculate_viscosity(T_upper_mantle)
        coef = self.params['thermal_expansivity'] * self.params['surface_gravity'] * self.params['density']
        Ra_internal = coef * delta_temp_internal * np.power(self.thickness, 3.) / (
            self.params['thermal_diffusivity'] * viscosity)
        return hard_coded_coef * np.power(Ra_internal, hard_coded_exponent)

    def calculate_temperature_base_stagnant_lid(self, T_upper_mantle, stagnant_lid_thickness):
        """
        Temperature at the base of the Stagnant Lid, equation (8) in Morschhauser et al 2011. 
        
        In previous numerical models and experiments, this temperature was found to be the temp.
        in which the viscosity had grown by an order of magnitude with respect to the convecting mantle.
        See Grasset and Parmentier 1998.
        """
        if stagnant_lid_thickness >= 0:
            theta = self.params['empirical_spherical_stagnant_lid_param']
            temperature_drop = theta * self.params['gas_constant'] * np.power(T_upper_mantle, 2.) / \
                           self.params['activation_energy']
        else:
            temperature_drop = 0

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

        thickness_convecting_mantle = self.thickness - stagnant_lid_thickness # - upper_boundary_layer_thickness \
 #                                     - lower_boundary_layer_thickness
        #print "Thickness of Convecting Mantle: ", thickness_convecting_mantle
        #print "Stagnant Lid Thickness: ", stagnant_lid_thickness
        #print "Layer Thickness: ", self.thickness
        #print "T_upper_mantle: ", T_upper_mantle
        g = self.params['surface_gravity']  # Is the Surface gravity the correct gravity here
        adiabatic_temperature_increase = self.params['thermal_expansivity'] * g * thickness_convecting_mantle * \
                                         T_upper_mantle / self.params['heat_capacity']
        temperature_base_mantle = T_upper_mantle + adiabatic_temperature_increase
        #print "Adiabatic Temp. Increase: ", adiabatic_temperature_increase
        #assert(temperature_base_mantle < T_cmb),"Temp. Base Mantle: {}, Temp. CMB: {}".format(temperature_base_mantle, T_cmb)

        return temperature_base_mantle

    def calculate_upper_boundary_layer_thickness(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Thickness of the boundary layer at the top of the convecting mantle, equation (12) in
        Morschhauser et al 2011. 

        The thickness of this boundary is calculated using the Nusselt-Rayleigh relation and
        critical boundary layer theory, see Turcotte and Schubert 2002.
        """
        beta = 1. / 3.
        Ra = self.calculate_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        upper_boundary_layer_thickness = (self.thickness - stagnant_lid_thickness) * np.power(self.params['critical_rayleigh_number'] / Ra, beta)
        #assert upper_boundary_layer_thickness<self.thickness
        assert upper_boundary_layer_thickness>0.0

        return upper_boundary_layer_thickness

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
        print T_cmb, temperature_base_mantle, delta_T
        assert(denominator>0.0)
        return np.power(numerator / denominator, 1. / 3.)

    def calculate_upper_heat_flux(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        """
        Heat flow from the convecting mantle into the stagnant lid, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle, stagnant_lid_thickness)
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

    def crustal_geometry(self, stagnant_lid_thickness=None):
        if stagnant_lid_thickness is not None:
            if self.crustal_thickness > stagnant_lid_thickness-100:
                print "RECYCLING CRUST!!!"
                crustal_thickness = np.min([self.crustal_thickness, stagnant_lid_thickness-100])
        else:
            crustal_thickness = self.crustal_thickness
            
        radius_crust = self.outer_radius - crustal_thickness
        volume_crust = 4./3.*np.pi*( np.power(self.outer_radius, 3.) - np.power(radius_crust, 3.) )
        return radius_crust, volume_crust

    def get_stagnant_lid_thermal_profile(self, T_upper_mantle, stagnant_lid_thickness, crustal_heating_rate,
                                         mantle_heating_rate):
        """
        Get the radial conductive temperature profile of the stagnant lid, equation (5) Morschhauser et al (2011).

        Will replace with the two layer eventually...
        :param T_upper_mantle:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        kc = self.params['crustal_thermal_conductivity']
        km = self.params['thermal_conductivity']
        #TODO: Replace with mantle and Crustal heating
        qc = crustal_heating_rate
        qm = mantle_heating_rate
        radius_surface = self.outer_radius
        radius_stagnant_lid, volume_crust = self.stagnant_lid_geometry(stagnant_lid_thickness)
        radius_crust, volume_crust = self.crustal_geometry()
        temperature_surface = self.params['surface_temperature']
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle, stagnant_lid_thickness)
        dT = temperature_base_stagnant_lid - temperature_surface

        delta_q = qc-qm
        k = kc/km

        alpha = np.power(radius_crust, 3.)*delta_q/(3*km)
        beta = np.power(radius_crust, 2)*(qm/km -qc/kc)/6 -alpha/radius_crust
        gamma = -(qm*np.power(radius_stagnant_lid, 2.))/(6*km) + alpha/radius_stagnant_lid + beta

        top = -qc*np.power(radius_surface, 2)/(6*kc)-gamma +dT
        bottom = k/radius_stagnant_lid +(1-k)/radius_crust -1/radius_surface

        c1 = top/bottom
        c2 = temperature_base_stagnant_lid -gamma - (k/radius_stagnant_lid +(1-k)/radius_crust)*c1
        m1 = alpha + k*c1
        m2 = beta + (1-k)*c1/radius_crust +c2

        crust_solution  = lambda r: (-qc*r*r/(6.*kc) + c1/r + c2) if((radius_surface  >= r)and(r >= radius_crust))  else 0
        mantle_solution = lambda r: (-qm*r*r/(6.*km) + m1/r + m2) if((radius_crust >= r)and(r >= radius_stagnant_lid))  else 0
        temperature_profile_as_function_of_radius = lambda r: crust_solution(r) + mantle_solution(r)
        gradient_at_base = -qm*radius_stagnant_lid/(3*km) -m1/(radius_stagnant_lid*radius_stagnant_lid)
        return temperature_profile_as_function_of_radius, gradient_at_base

    def calculate_thermal_gradient_base_stagnat_lid(self, T_upper_mantle, stagnant_lid_thickness, crustal_heating_rate,
                                         mantle_heating_rate):
        """
        Calculate the thermal gradient at the base of the stagnant lid by solving the radial heat conduction
        equation in the stagnant lid, equation (5), and evaluating its derivative at the base of the stagnant.
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        temp_profile, gradient_at_base =self.get_stagnant_lid_thermal_profile(T_upper_mantle, stagnant_lid_thickness, crustal_heating_rate,
                                         mantle_heating_rate)
        radius_stagnant_lid = self.outer_radius - stagnant_lid_thickness
        #print "Gradient at the base of the Stagnant Lid:", gradient_at_base, derivative(temp_profile , radius_stagnant_lid, dx=1e-1 )
        return gradient_at_base

    def get_rate_of_stagnant_lid_growth(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, crustal_heating_rate,
                                         mantle_heating_rate):
        """
        The growth of the stagnant lid is determined by the energy balance at the base of the stagnant lid,
        equation (4) Morschhauser et al (2011).

        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        temperature_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle, stagnant_lid_thickness)
        
        delta_T = T_upper_mantle - temperature_base_stagnant_lid
        if delta_T != 0.0:
            lhs_coef = self.params['density']*self.params['heat_capacity']*delta_T
            flux_thermal_gradient = self.params['thermal_conductivity']*self.calculate_thermal_gradient_base_stagnat_lid(
            T_upper_mantle, stagnant_lid_thickness, crustal_heating_rate, mantle_heating_rate)
            upper_heat_flux = self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
            dDlid_dt = (-upper_heat_flux - flux_thermal_gradient)/lhs_coef
        else:
            dDlid_dt = 0.0
        return dDlid_dt

    def energy_conservation_mantle(self, T_upper_mantle, T_cmb, stagnant_lid_thickness, mantle_heating_rate):
        """
        Equation (1) in Morschhauser et al (2001),the energy conservation equation in the mantle to
        be solved to determine the thermal evolution of the planet.
        :param T_upper_mantle:
        :param T_cmb:
        :param stagnant_lid_thickness:
        :param mantle_heat_production:
        :return:
        """
        if stagnant_lid_thickness<0:
            stagnant_lid_thickness=0.0
        radius_stagnant_lid, volume_stagnant_lid = self.stagnant_lid_geometry(stagnant_lid_thickness)
        surface_area_base_stagnant_lid = 4.*np.pi*np.power(radius_stagnant_lid, 2)
        volume_core = 4./3.*np.pi*np.power(self.inner_radius, 3)
        volume_convecting_mantle = volume_stagnant_lid - volume_core
        lhs_coef = self.effective_heat_capacity(volume_convecting_mantle)
        upper_heat_flux= self.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        lower_heat_flux = self.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        crust_radius, crust_volume = self.crustal_geometry()
        dTm_dt = (-upper_heat_flux*surface_area_base_stagnant_lid + lower_heat_flux*self.inner_surface_area +
                  mantle_heating_rate*volume_convecting_mantle)/lhs_coef
        return dTm_dt

    def set_melt_fraction(self):
        '''
        Fractionation due to batch-melting, equation (12) from Hauck et al 2004.
        :param melt_fraction:
        :return:
        '''
        radius_crust, volume_crust = self.crustal_geometry()
        volume_mantle = self.volume
        self.melt_fraction = volume_crust/volume_mantle
        #TODO: Need to account for the lower concentration in the mantle due to the enriched crust.
        enhancement_factor = self.radiogenic_heating_model.batch_melting_fractionization(self.melt_fraction)
        self.crust_radiogenic_fraction = enhancement_factor
        self.mantle_radiogenic_fraction = 1.0 - enhancement_factor*(volume_crust*self.params['crustal_density'])/(self.volume*self.params['density'])

    def mantle_heating_rate(self, time):
        return self.mantle_radiogenic_fraction * self.params['density']*self.radiogenic_heating_model.heat_production(time)

    def crustal_heating_rate(self, time):
        return self.crust_radiogenic_fraction * self.params['crustal_density']*self.radiogenic_heating_model.heat_production(time)

    def core_energy_balance(self, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        lhs = rho_core*core_heat_capacity*4./3.*np.pi*np.power(self.inner_radius, 3.)
        lower_heat_flux = self.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness)
        dTc_dt = -lower_heat_flux*self.inner_surface_area/lhs
        return dTc_dt

    def energy_balance(self, time, T_upper_mantle, T_cmb, stagnant_lid_thickness):
        self.time = time
        mantle_heating_rate = self.mantle_heating_rate(time)
        print "Mantle Heating:", mantle_heating_rate
        crustal_heating_rate = self.crustal_heating_rate(time)
        print "Crustal Heating:", crustal_heating_rate
        dTm_dt = self.energy_conservation_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness, mantle_heating_rate)
        print "dTm_dt :", dTm_dt
        dDlid_dt = self.get_rate_of_stagnant_lid_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness,
                                                        crustal_heating_rate, mantle_heating_rate)
        print "dDlid_dt :", dDlid_dt
        dTc_dt = self.core_energy_balance(T_upper_mantle, T_cmb, stagnant_lid_thickness )
        print "dTc_dt:", dTc_dt
        return np.array([dTm_dt, dTc_dt, dDlid_dt])

    def integrate(self):
        def ODE(y, t):
           print "T_upper_mantle: ", y[0]
           print "T_cmb         : ", y[1]
           print "Stagnant Lid Thickness: ", y[2]
           return self.energy_balance(t, y[0], y[1], y[2])

        times = np.linspace(0., Julian_year * 4.5e9, 1000)
        solution = integrate.odeint( ODE, self.initial_conditions, times)
        return times, solution

if __name__ == "__main__":
    radius_planet = 2440.e3
    radius_cmb = 2020.e3
    crustal_thickness = 120.e3
    initial_Tm = 1700.
    initial_Tcmb = 2000.
    initial_Dlid= 100.e3


    merc = MantleLayer(radius_cmb, radius_planet, crustal_thickness, mantle_params, WD94, initial_Tcmb, initial_Tm, initial_Dlid)
    times, solution = merc.integrate()
    #######

    T_upper_mantle = initial_Tm
    T_cmb = initial_Tcmb
    stagnant_lid_thickness = initial_Dlid
    crustal_heating_rate = 0.0
    mantle_heating_rate = 0.0
    tm = []
    tc = []
    dTm_dt = []
    dDlid_dt = []
    dTc_dt = []
    Ra = []
    mu = []
    Tbsl = []
    Tbm = []
    qu = []
    ql = []
    dubl = []
    dlbl =[]

    # for ii, T_upper_mantle in enumerate(np.linspace(2000, 1000, 1000)):
    #     T_cmb = T_upper_mantle + 300
    #     tm.append( T_upper_mantle)
    #     tc.append(T_cmb)
    #     dTm_dt.append(merc.energy_conservation_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness, mantle_heating_rate))
    #     dDlid_dt.append(merc.get_rate_of_stagnant_lid_growth(T_upper_mantle, T_cmb, stagnant_lid_thickness, crustal_heating_rate, mantle_heating_rate))
    #     dTc_dt.append( merc.core_energy_balance(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     Ra.append(merc.calculate_rayleigh_number(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     mu.append(merc.calculate_viscosity(T_upper_mantle))
    #     Tbsl.append(merc.calculate_temperature_base_stagnant_lid(T_upper_mantle, stagnant_lid_thickness))
    #     Tbm.append(merc.calculate_temperature_base_mantle(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     qu.append(merc.calculate_upper_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     ql.append(merc.calculate_lower_heat_flux(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     dubl.append(merc.calculate_upper_boundary_layer_thickness(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #     dlbl.append(merc.calculate_lower_boundary_layer_thickness(T_upper_mantle, T_cmb, stagnant_lid_thickness))
    #
    # plt.figure(1)
    # plt.plot(tm, dTm_dt, label = 'dTm_dt')
    # plt.plot(tm, dTc_dt, label = 'dTc_dt')
    # plt.legend()
    #
    # plt.figure(2)
    # plt.plot(tm, dDlid_dt, label= 'dDlid')
    # plt.legend()
    #
    # plt.figure(3)
    # plt.semilogy(tm, Ra, label= 'Rayleigh Number')
    # plt.axhline(5*10^3)
    # plt.legend()
    #
    # plt.figure(4)
    # plt.semilogy(tm, mu, label='Viscosity')
    # plt.legend()
    #
    # plt.figure(5)
    # plt.plot(tm, Tbsl, label="Temperature Base of Stagnant Lid")
    # plt.plot(tm, Tbm,  label="Temperateure Base Of Mantle")
    # plt.legend()
    #
    # plt.figure(6)
    # plt.plot(tm, qu, label="Upper Heat Flux")
    # plt.plot(tm, ql,  label="Lower Heat Flux")
    # plt.legend()
    #
    # plt.figure(7)
    # plt.plot(tm, dubl, label="Upper Boundary Layer Thickness")
    # plt.plot(tm, dlbl,  label="LLower boundary layer Thickness")
    # plt.legend()
    #
    # crustal_heating_rate = merc.mantle_heating_rate(0)
    # mantle_heating_rate  = merc.crustal_heating_rate(0)
    #
    # plt.figure(8)
    # r = np.linspace(radius_cmb, radius_planet, 100)
    # p = merc.convert_radius_to_hydrostatic_pressure(r)
    # tprofile, _ = merc.get_stagnant_lid_thermal_profile(initial_Tm, initial_Dlid, crustal_heating_rate, mantle_heating_rate)
    # temp_in_lid = []
    # for ri in r:
    #     temp_in_lid.append(tprofile(ri))
    # mercury_solidus = melt_model.mantle_solidus(p, crustal_thickness, radius_cmb, radius_planet)
    # peridotite_liquidus = melt_model.peridotite_liquidus(p)
    # peridotite_solidus = melt_model.peridotite_liquidus(p)
    # plt.plot(r, mercury_solidus,label='Mercury Solidus')
    # plt.plot(r, peridotite_liquidus,label='Peridotite Liquidus')
    # plt.plot(r, peridotite_solidus,label='Peridotite Solidus')
    # plt.plot(r, temp_in_lid,label='Initial Profile in Lid' )
    # plt.legend()
    # plt.show()

    #print times, solution
    plt.figure()
    plt.plot(times,solution[:, 0])
    plt.plot(times,solution[:, 1])
    ##plt.plot(times,solution[:, 2])
    plt.show()
