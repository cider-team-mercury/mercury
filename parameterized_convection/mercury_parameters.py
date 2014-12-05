import numpy as np

# ------------------------------------------------------ #
# - Fixed Parameters from Grott et al 2011 for Mercury - #
# ------------------------------------------------------ #

Radius  = 2440.0 # - km
gravity = 3.7    # - m s^-2

rho_crust  = 2800.0 # - kg m^-3
rho_mantle = 3400.0 # - kg m^-3
rho_core   = 7200.0 # - kg m^-3
rho_depleted_mantle = 3450.0 # - kg m^-3

magma_heat_capacity  = 1000.0 # - J Kg^-1 K^-1
mantle_heat_capacity = 1212.0 # - J Kg^-1 K^-1
core_heat_capacity   = 465.0  # - J Kg^-1 K^-1 

epsilon_mantle = 1.0 # - Ratio of mean and upper mantle Temp
epsilon_core   = 1.1 # - Ratio of mean and upper core Temp

R = 8.3144 # - J K^-1 mol^-1 - Gas Constant
A = 3.0e5 # - J mol^-1 - Activation Energy

T_ref  = 1600.0 # - K
T_surf = 440.0  # - K

k_regolith = 0.2 # - W m^-1 K^-1 - Regolith Thermal Conductivity
k_mantle   = 4.0 # - W m^-1 K^-1 - Mantle Thermal Conductivity

mantle_diffusivity = 1.0e-6 # - m^s s^-1

alpha_mantle = 2.0e-5 # - K^-1
alpha_core   = 3.0e-5 # - K^-1

latent_heat_melting_crust       = 6.0e-5   # - J kg^-1
# It is unclear here what is used in Grott et al 2011
latent_heat_solidification_core = 2.5e-5 # - J kg^-1
grav_energy_release             = 2.5e-5 # - J kg^-1

Ra_crit = 450.0 # - Critical Rayleigh Number
convection_speed_scale = 2.0e-12 # m s^-1

initial_stagnant_lid_thickness = 50.0 # - km
primordial_crust_thickness = 5.0 # - km
crustal_enrichment = 4.0 # Enrichment of radioactive elements in the crust with respect to the undepleted mantle
fraction_exctractable_crust = 0.4 
Theta = 2.9 # Empirical Coefficient for Stagnant Lid Convection from Grasset and Partentier
# ------------------------------------------------------- #
# - Varied Parameters from Grott et al 2011 for Mercury - #
# ------------------------------------------------------- #
core_radius = 2050.0 # - km - 1840 km to 2050 km
T_upper_mantle_initial = 2000.0 # - K - 1650 K to 2000 K
T_core_excess = 300.0 # - K - 0 K to 300 K
k_crust = 4.0 # - W m^-1 K^-1 - 1.5 to 4
mantle_viscosity_ref = 1.0e22 # - Pa s - 10^19 to 10^22
regolith_thickness = 5.0e3 # - m - 0 to 5*10^3 m 
mantle_differentiation_volume_change = 5.0 # - % - 0% to 5%

core_params = { 
    'rho' : rho_core,
    'c'   : core_heat_capacity,
    'L+Eg': latent_heat_solidification_core+grav_energy_release,
    'mu' : epsilon_core
    }

mantle_params = {
    'density' : rho_mantle,
    'heat_capacity'   : mantle_heat_capacity,
    'epsilon' : epsilon_mantle,
    'gas_constant' : R,
    'activation_energy' : A,
    'empirical_spherical_stagnant_lid_param' : Theta,
    'thermal_expansivity' : alpha_mantle,
    'surface_gravity' : gravity, 
    'reference_viscosity' : mantle_viscosity_ref,
    'reference_temperature': T_ref,
    'surface_temperature' : T_surf,
    'thermal_diffusivity' : mantle_diffusivity,
    'factor_pressure_dependent_viscosity' : 1.0, # Not sure what to put here for now.
    'thermal_conductivity' : k_mantle,
    'mantle_convection_speed_scale': convection_speed_scale,
    'critical_rayleigh_number': Ra_crit,
    'latent_heat_melting_crust': latent_heat_melting_crust,
    'magma_heat_capacity' : magma_heat_capacity,
    'crustal_density' : rho_crust
    }

