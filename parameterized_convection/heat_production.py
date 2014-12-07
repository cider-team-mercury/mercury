import numpy as np
from scipy.constants import pi, N_A, Julian_year, gram


class radioactive_species(object):
    def __init__(self, heat_release, half_life, bulk_concentration, molar_mass, partition_coefficient=None, ppm_given=True):
        """
        Assumes that the concentrations are given in ppm. It turns out that most close-packed oxides and
        silicates have a mean atomic mass close to 20, here we use a average Mantle value of 21.1 g
        followingPoirier (1991).
        Mg2SiO4 = 21.13 g
        MgSiO3 = 20.12 g
        MgO = 20.15 g

        :param heat_release:
        :param half_life:
        :param concentration:
        :param partition_coefficient:
        :param ppm_given:
        :return:
        """
        self.mean_mantle_atomic_mass = 21.1*gram # kg
        self.species_molar_mass = molar_mass
        self.half_life = half_life * Julian_year  # Convert from years to seconds
        self.heat_release = heat_release
        if ppm_given:
            self.bulk_concentration = bulk_concentration  * self.species_molar_mass/(1.e6*self.mean_mantle_atomic_mass) # convert from ppm to kg/kg
        else:
            self.bulk_concentration = bulk_concentration
        self.partition_coefficient = partition_coefficient

    def heat_generation_rate(self,time):
        return self.bulk_concentration*self.heat_release*np.exp(-np.log(2.)*time/self.half_life)

    def liquid_partioning(self, F):
        """
        Calculate the concentration of heat producing elements in the liquid phase
        from the depth dependent melt fraction, F.
        :param F:
        :return:
        """
        assert self.bulk_concentration is not None, "Bulk Concentration not set."
        assert 0 <= F <= 1, "Depth Dependent Melt Fraction is not between 0 and 1"
        assert self.partition_coefficient is not None, "Partition coefficient not set."
        liquid_concentration = self.bulk_concentration/F*[1-np.power(1-F, 1/self.partition_coefficient)]


class radiogenic_heating(object):
    def __init__(self,uranium_params,thorium_params,potassium_params, model_name=None):
        self.uranium_molar_mass = 238.03*gram
        self.thorium_molar_mass = 232.04*gram
        self.potassium_molar_mass = 39.10*gram
        self.model_name = model_name

        self.uranium = radioactive_species(molar_mass=self.uranium_molar_mass, **uranium_params)
        self.thorium = radioactive_species(molar_mass=self.thorium_molar_mass, **thorium_params)
        self.potassium = radioactive_species(molar_mass=self.potassium_molar_mass, **potassium_params)

    def heat_production(self, time):
        return self.uranium.heat_generation_rate(time) + self.thorium.heat_generation_rate(time) + \
                self.potassium.heat_generation_rate(time)


# ------------------------------------------------------------- #
# - Partition coefficients
# ------------------------------------------------------------- #
# The Partition coefficients are similar for all relevent heat producing
# elements, Beattie (1993).

# From Hart and Dunn (1993), and Hauri et al (1994):
D_clinopyroxene = 0.01
D_garnet        = 0.004

# Mars composition, from Wanke and Dreibus (1994):
D_Wanke_Dreibus_Mars = 0.002

# ------------------------------------------------------------- #
# -  Radiogenic Abundances
# ------------------------------------------------------------- #
# Mercury Surface Composition from Peplowski et al (2011), see supplement:
C_Potassium_Peplowski = 1150. # ppm
C_Uranium238_Peplowski  = 220e-3 # ppm
C_Thorium232_Peplowski  = 90e-3 # ppm

# "Bulk Silicate Earth" from McDonough and Sun (1995):
C_Potassium_BSE = 240. # ppm
C_Thorium_BSE  = 20.3e-3 # ppm
C_Uranium_BSE  = 79.5e-3 # ppm

# CI Carbonaceous Chondrites  from McDonough and Sun (1995):
C_Potassium_CI= 550. # ppm
C_Uranium_CI  = 7.4e-3 # ppm
C_Thorium_CI  = 29.e-3 # ppm

# [High] Mars Composition, Lodders and Fegley (1997):
C_Potassium_Lodders  = 920. # ppm
C_Thorium_Lodders  = 55.e-3 # ppm
C_Uranium_Lodders = 16.e-3 # ppm

# - See Hauck and Philips (2002) for Comparison.
# [LOW] Mars Composition, Wanke and Dreibus (1994):
C_Potassium_Wanke  = 305. # ppm
C_Thorium_Wanke  = 56.e-3 # ppm
C_Uranium_Wanke = 16.e-3 # ppm

# Earth Composition, Turcotte and Schubert (1982):
C_Uranium235_TS = 30.8e-9 # kg kg^-1
C_Uranium238_TS = 0.22e-9 # kg kg^-1
C_Thorium232_TS = 124.e-9 # kg kg^-1
C_Potassium40_TS = 36.9e-9 # kg kg^-1
# "Natural Uranium, Thorium, Potasium"
# Uranium is 99.28% U238, 0.71% U235
# Thorium is 100% Th232
# Potasium is 0.0119% K40
C_Potasium_TS = 31.0e-5 # kg kg^-1
C_Uranium_TS = 31.0e-9 # kg kg^-1
C_Thorium_TS = C_Thorium232_TS # kg kg^-1


# ------------------------------------------------------------- #
# - Heat Production and Half Lives from Turcoette and Schubert 2002:
# ------------------------------------------------------------- #
# Heat Production Rates:
H_Uranium238 = 9.46e-5 # W kg^-1
H_Uranium235 = 9.69e-4 # W kg^-1
H_Thorium232 = 2.64e-5 # W kg^-1
H_Potasium40 = 2.92e-5 # W kg^-1
# "Natural Uranium, Thorium, Potasium"
# Uranium is 99.28% U238, 0.71% U235
# Thorium is 100% Th232
# Potasium is 0.0119% K40
H_Uranium = 9.81e-5 # W kg^-1
H_Thorium = H_Thorium232 # W kg^-1
H_Potasium = 3.48e-9 # W kg^-1

# Half Lives:
T_Uranium238 = 4.47e9 #years
T_Uranium235 = 7.04e8 #years
T_Thorium232 = 1.40e10 #years
T_Potasium40 = 1.25e9 #years

# ------------------------------------------------------------------------ #
# Turcotte and Schubert Models:
TS82_uranium_params = { 'heat_release': H_Uranium,
                        'half_life'   : T_Uranium238,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Uranium_TS,
                        'ppm_given' : False
                        }

TS82_potassium_params = { 'heat_release': H_Potasium,
                        'half_life'   : T_Potasium40,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Potasium_TS,
                        'ppm_given' : False
                        }

TS82_thorium_params = { 'heat_release': H_Thorium,
                          'half_life'   : T_Thorium232,
                          'partition_coefficient' : D_Wanke_Dreibus_Mars,
                          'bulk_concentration' : C_Thorium_TS,
                          'ppm_given' : False
                         }
TS82 = radiogenic_heating(TS82_uranium_params, TS82_thorium_params, TS82_potassium_params, 'Turcotte and Schubert [1984] ')

# Bulk Silicate Earth Models:
CI_uranium_params = { 'heat_release': H_Uranium238,
                        'half_life'   : T_Uranium238,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Uranium_CI
                        }

CI_potassium_params = { 'heat_release': H_Potasium,
                        'half_life'   : T_Potasium40,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Potassium_CI
                        }

CI_thorium_params = { 'heat_release': H_Thorium,
                          'half_life'   : T_Thorium232,
                          'partition_coefficient' : D_Wanke_Dreibus_Mars,
                          'bulk_concentration' : C_Thorium_CI
                         }
CI = radiogenic_heating(CI_uranium_params, CI_thorium_params, CI_potassium_params, '"CI Carbonaceous Chondrites" [1995]')

# Bulk Silicate Earth Models:
BSE_uranium_params = { 'heat_release': H_Uranium,
                        'half_life'   : T_Uranium238,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Uranium_BSE
                        }

BSE_potassium_params = { 'heat_release': H_Potasium,
                        'half_life'   : T_Potasium40,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Potassium_BSE
                        }

BSE_thorium_params = { 'heat_release': H_Thorium,
                          'half_life'   : T_Thorium232,
                          'partition_coefficient' : D_Wanke_Dreibus_Mars,
                          'bulk_concentration' : C_Thorium_BSE
                         }
BSE = radiogenic_heating(BSE_uranium_params, BSE_thorium_params, BSE_potassium_params, '"Bulk Silicate Earth" [1995]')

# Wanke and Dreibus Models:
WD94_uranium_params = { 'heat_release': H_Uranium238,
                        'half_life'   : T_Uranium238,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Uranium_Wanke
                        }

WD94_potassium_params = { 'heat_release': H_Potasium,
                        'half_life'   : T_Potasium40,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Potassium_Wanke
                        }

WD94_thorium_params = { 'heat_release': H_Thorium,
                          'half_life'   : T_Thorium232,
                          'partition_coefficient' : D_Wanke_Dreibus_Mars,
                          'bulk_concentration' : C_Thorium_Wanke
                         }

WD94 = radiogenic_heating(WD94_uranium_params, WD94_thorium_params, WD94_potassium_params, 'Wanke and Dreibus [1994]')

# Lodders and Fegley Models:
LF97_uranium_params = { 'heat_release': H_Uranium238,
                        'half_life'   : T_Uranium238,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Uranium_Lodders
                        }

LF97_potassium_params = { 'heat_release': H_Potasium,
                        'half_life'   : T_Potasium40,
                        'partition_coefficient' : D_Wanke_Dreibus_Mars,
                        'bulk_concentration' : C_Potassium_Lodders
                        }

LF97_thorium_params = { 'heat_release': H_Thorium,
                          'half_life'   : T_Thorium232,
                          'partition_coefficient' : D_Wanke_Dreibus_Mars,
                          'bulk_concentration' : C_Thorium_Lodders
                         }
LF97 = radiogenic_heating(LF97_uranium_params, LF97_thorium_params, LF97_potassium_params, 'Lodders and Fegley [1997]')

def schubert_spoon_model(time):
    mtime = time.max()
    return 1.7e-7*np.exp(-1.38e-17 *(time))/3527


import matplotlib.pyplot as plt
time = np.linspace(0, Julian_year*4.5e9, 1000)
models = [WD94, LF97, CI, TS82]
fig = plt.figure(figsize=[20,10])
for ii, model in enumerate(models):
    #plt.subplot(1,len(models),ii)
    heat = model.heat_production(time)
    plt.plot(time/(1.e9*Julian_year), heat, label=model.model_name)
    plt.legend()
    plt.xlabel(r'Time [Ga]')
    plt.ylabel(r'Heat Production [W/kg]')
    #plt.plot(time/(1.e9*Julian_year), schubert_spoon_model(time))
    plt.title('Heat Production Comparison')
#plt.gca().invert_xaxis()
plt.show()

