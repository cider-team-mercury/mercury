import numpy as np
from planetary_energetics import *

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
A = 3.0*10^5 # - J mol^-1 - Activation Energy

T_ref  = 1600.0 # - K
T_surf = 440.0  # - K

k_regolith = 0.2 # - W m^-1 K^-1 - Regolith Thermal Conductivity
k_mantle   = 4.0   # - W m^-1 K^-1 - Mantle Thermal Conductivity

mantle_diffusivity = 1.0*10^-6 # - m^s s^-1

alpha_mantle = 2.0*10^-5 # - K^-1
alpha_core   = 3.0*10^-5 # - K^-1

latent_heat_melting_crust       = 6.0*10^-5   # - J kg^-1
# It is unclear here what is used in Grott et al 2011
latent_heat_solidification_core = 2.5*10^-5 # - J kg^-1
grav_energy_release             = 2.5*10^-5 # - J kg^-1

Ra_crit = 450.0 # - Critical Rayleigh Number
convection_speed_scale = 2.0*10^-12 # m s^-1

initial_stagnant_lid_thickness = 50.0 # - km
primordial_crust_thickness = 5.0 # - km
crustal_enrichment = 4.0 # Enrichment of radioactive elements in the crust with respect to the undepleted mantle
fraction_exctractable_crust = 0.4 

# ------------------------------------------------------- #
# - Varied Parameters from Grott et al 2011 for Mercury - #
# ------------------------------------------------------- #
core_radius = 2050.0 # - km - 1840 km to 2050 km
T_upper_mantle_initial = 2000.0 # - K - 1650 K to 2000 K
T_core_excess = 300.0 # - K - 0 K to 300 K
k_crust = 4.0 # - W m^-1 K^-1 - 1.5 to 4
viscosity_ref = 1.0*10^22 # - Pa s - 10^19 to 10^22
regolith_thickness = 5.0*10^3 # - m - 0 to 5*10^3 m 
mantle_differentiation_volume_change = 5.0 # - % - 0% to 5%
# ------------------------------------------------------- #

# Will Try and Modularize this later
#core_params = {
#            'regime' : 'Temp. Dependent Viscosity',
#            'epsilon' : epsilon_mantle
#            'Ra_c' : Ra_crit, #critical Rayleigh number
#            'rho' : rho_core, # density; kg/m^3
#            'H_0' : 0., #heat production W/m^3
#            'decay_constant' : 1.38e-17, # some half-life
#            'nu_0' : viscosity_ref, #kinematic viscosity m^2/s
#            'A_0' : A, # Activation Energy
#            'R'   : R, # Gas Constant
#            'k' : , #thermal conductivity W/m/K
#            'kappa' : 1.e-6, #thermal diffusivity m^2/s
#            'alpha' : 2.e-5, #thermal expansion 1/K
#            'g' : 3., #acceleration of gravity m/s^2
#            'beta' : 1./3, #parameter Nu=Ra^(1/beta)
#            }


###!!!!####
# - Note that the volume and Area of the mantle are currently incorect since
# - we have neglected the crust
V_mercury = 4./3.*np.pi*radius**3      # - Volume of Mercury
V_core    = 4./3.*np.pi*core_radius**3 # - Volume of Core
V_mantle  = V_mercury-V_core           # - Volume of Mantle
A_core    = 4.*np.pi*core_radius**2    # - Surface Area of Core
A_mantle  = 4.*np.pi*radius**2         # - Surface Area of Mantle



core_params = { 
    'rho' : rho_core,
    'c'   : core_heat_capacity,
    'V'   : V_core,
    'A'   : A_core,
    'L+Eg': latent_heat_solidification_core+grav_energy_release,
    'epsilon' : epsilon_core
    }

mantle_params = {
    'rho' : rho_mantle,
    'c'   : mantle_heat_capacity,
    'V'   : V_mantle,
    'A'   : A_mantle,
    'epsilon' : epsilon_mantle
    }

def core_energy_balance(core_flux, delta_core_radius, p):
    return -core_flux*p['A']/(p['rho']*p['c']*p['V']*p['epsilon']-p['L+Eg']*p['rho']*p['A']*delta_core_radius)

def mantle_energy(delta_crustal_thickness, p):
    # Tm-Tl  - Temperature of the upper mantle minus Temperature at the base of the 
                    # - stagnant lid
    coef_LHS = rho_mantle*mantle_heat_capacity*V_mantle*epsilon_mantle*(1+St)
    coef_crust = (rho_crust*latent_heat_melting_crust + rho_crust*magma_heat_capacity*(T_mantle-T_lid)
    return -( (mantle_flux + coef_crust*delta_crustal_thickness)*A_mantle + core_flux*A_core +\
                heating_rate*V_mantle )/coef_LHS

