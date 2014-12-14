import numpy as np
from scipy.constants import pi, N_A, Julian_year, gram
# - Citations:## Beattie, Paul. "The generation of uranium series disequilibria by partial melting of
# spinel peridotite: constraints from partitioning studies." EPSL 117.3 (1993): 379-391.
#
#  Hart, Stanley R., and Todd Dunn. "Experimental melt partitioning of 24 trace elements."
#  Contributions to Mineralogy and Petrology 113.1 (1993): 1-8.
#
#  Hauri, Erik H., Thomas P. Wagner, and Timothy L. Grove. "Experimental and natural partitioning
#  of Th, U, Pb and other trace elements between garnet, clinopyroxene and basaltic melts."
#  Chemical Geology 117.1 (1994): 149-166.
#
#  Wanke, Heinrich, Gerlind Dreibus, and I. P. Wright. "Chemistry and accretion history of Mars."
#  Philosophical Transactions of the Royal Society of London. Physical and Engineering
#  Sciences 349.1690 (1994): 285-293.
#
#  Peplowski, Patrick N., et al. "Radioactive elements on Mercurys surface from MESSENGER:
#  Implications for the planets formation and evolution." Science 333.6051 (2011): 1850-1852.
#
#  Turcotte, Donald L., and Gerald Schubert. Geodynamics. Cambridge University Press, 2014.
#
#  Lodders, K., and B. Fegley Jr. "An oxygen isotope model for the composition of Mars."
#  Icarus 126.2 (1997): 373-394.
#
#  McDonough, William F., and S-S. Sun. "The composition of the Earth." Chemical geology
#  120.3 (1995): 223-253.
#
#  Hauck II, Steven A., et al. "Internal and tectonic evolution of Mercury." Earth and
#  Planetary Science Letters 222.3 (2004): 713-728.


class radioactive_species(object):
    def __init__(self, heat_release, half_life, bulk_concentration, partition_coefficient=None):
        """
        Assumes that the concentrations are given in ppm, WHICH IS WEIGHT FRACTION!
        :param heat_release:
        :param half_life:
        :param concentration:
        :param partition_coefficient:
        :param ppm_given:
        :return:
        """
        self.half_life = half_life * Julian_year  # Convert from years to seconds
        self.heat_release = heat_release
        self.four_and_a_half_Gyr = 3.5*Julian_year*1.e9
        self.bulk_concentration = bulk_concentration*1.e-6 # ppm
        self.initial_bulk_concentration = self.bulk_concentration * np.exp(self.four_and_a_half_Gyr * np.log(2.) / self.half_life)
        self.partition_coefficient = partition_coefficient

    def heat_generation_rate(self, time):
        return self.heat_release * self.initial_bulk_concentration * np.exp(-np.log(2.) * (time) / self.half_life)

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
        liquid_concentration = (self.bulk_concentration / F) * [1 - np.power(1 - F, 1 / self.partition_coefficient)]
        return liquid_concentration


class radiogenic_heating(object):
    def __init__(self, uranium_params, thorium_params, potassium_params, model_name=None):
        self.model_name = model_name
        U238fraction = 0.9928
        U235fraction = 0.0071
        K40fraction  = 0.000119
        Th232fraction = 1.0
        # From Hauck et al 2004
        self.bulk_distribution_coefficient = 0.1
        natural_uranium_concentration = uranium_params.pop('bulk_concentration')
        #TODO FINISH THIS.
        self.uranium235 = radioactive_species(heat_release = uranium_params.pop('heat_release_235'),
                                              half_life = uranium_params.pop('half_life_235'),
                                              bulk_concentration = U235fraction*natural_uranium_concentration )

        self.uranium238 = radioactive_species(heat_release = uranium_params.pop('heat_release_238'),
                                              half_life = uranium_params.pop('half_life_238'),
                                              bulk_concentration = U238fraction*natural_uranium_concentration  )
        self.thorium = radioactive_species(**thorium_params)
        self.potassium = radioactive_species(**potassium_params)

    def heat_production(self, time):
        total_heat = self.uranium235.heat_generation_rate(time) + self.uranium238.heat_generation_rate(time) + \
                     self.thorium.heat_generation_rate(time) + 0.000119*self.potassium.heat_generation_rate(time)
        return total_heat

    def batch_melting_fractionization(self, melt_fraction):
        '''
        Fractionation due to batch-melting, equation (12) from Hauck et al 2004.
        :param melt_fraction:
        :return:
        '''
        concentration_melt = 1/(melt_fraction+self.bulk_distribution_coefficient*(1-melt_fraction))
        return concentration_melt

# ------------------------------------------------------------- #
# - Partition coefficients
# ------------------------------------------------------------- #
# The Partition coefficients are similar for all relevent heat producing
# elements, Beattie (1993).

# From Hart and Dunn (1993), and Hauri et al (1994):
D_clinopyroxene = 0.01
D_garnet = 0.004

# Mars composition, from Wanke and Dreibus (1994):
D_Wanke_Dreibus_Mars = 0.002

# ------------------------------------------------------------- #
# -  Radiogenic Abundances
# ------------------------------------------------------------- #
# Mercury Surface Composition from Peplowski et al (2011), see supplement:
C_Potassium_Peplowski = 1150.  # ppm
C_Uranium238_Peplowski = 220e-3  # ppm
C_Thorium232_Peplowski = 90e-3  # ppm

# "Bulk Silicate Earth" from McDonough and Sun (1995):
C_Potassium_BSE = 240.  # ppm
C_Thorium_BSE = 20.3e-3  # ppm
C_Uranium_BSE = 79.5e-3  # ppm

# CI Carbonaceous Chondrites  from McDonough and Sun (1995):
C_Potassium_CI = 550.  # ppm
C_Uranium_CI = 7.4e-3  # ppm
C_Thorium_CI = 29.e-3  # ppm

# [High] Mars Composition, Lodders and Fegley (1997):
C_Potassium_Lodders = 920.  # ppm
C_Thorium_Lodders = 55.e-3  # ppm
C_Uranium_Lodders = 16.e-3  # ppm

# - See Hauck and Philips (2002) for Comparison.
# [LOW] Mars Composition, Wanke and Dreibus (1994):
C_Potassium_Wanke = 305.  # ppm
C_Thorium_Wanke = 56.e-3  # ppm
C_Uranium_Wanke = 16.e-3  # ppm

# Earth Composition, Turcotte and Schubert (1982):
C_Uranium235_TS = 30.8e-9  # kg kg^-1
C_Uranium238_TS = 0.22e-9  # kg kg^-1
C_Thorium232_TS = 124.e-9  # kg kg^-1
C_Potassium40_TS = 36.9e-9  # kg kg^-1
# "Natural Uranium, Thorium, Potasium"
# Uranium is 99.28% U238, 0.71% U235
# Thorium is 100% Th232
# Potasium is 0.0119% K40
C_Potasium_TS = 31.0e-5  # kg kg^-1
C_Uranium_TS = 31.0e-9  # kg kg^-1
C_Thorium_TS = C_Thorium232_TS  # kg kg^-1


# ------------------------------------------------------------- #
# - Heat Production and Half Lives from Turcoette and Schubert 2002:
# ------------------------------------------------------------- #
# Heat Production Rates:
H_Uranium238 = 9.46e-5  # W kg^-1
H_Uranium235 = 5.69e-4  # W kg^-1
H_Thorium232 = 2.64e-5  # W kg^-1
H_Potasium40 = 2.92e-5  # W kg^-1
# "Natural Uranium, Thorium, Potasium"
# Uranium is 99.28% U238, 0.71% U235
# Thorium is 100% Th232
# Potasium is 0.0119% K40
H_Uranium = 9.81e-5  # W kg^-1
H_Thorium = H_Thorium232  # W kg^-1
H_Potasium = 3.48e-9  # W kg^-1

# Half Lives:
T_Uranium238 = 4.47e9  #years
T_Uranium235 = 7.04e8  #years
T_Thorium232 = 1.40e10  #years
T_Potasium40 = 1.25e9  #years
# ------------------------------------------------------------------------ #
# Turcotte and Schubert Models:
TS82_uranium_params = {'heat_release_235': H_Uranium235,
                       'heat_release_238': H_Uranium238,
                       'half_life_235': T_Uranium235,
                       'half_life_238': T_Uranium238,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Uranium_TS,
}

TS82_potassium_params = {'heat_release': H_Potasium40,
                         'half_life': T_Potasium40,
                         'partition_coefficient': D_Wanke_Dreibus_Mars,
                         'bulk_concentration': C_Potasium_TS,
}

TS82_thorium_params = {'heat_release': H_Thorium,
                       'half_life': T_Thorium232,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Thorium_TS,
}
TS82 = radiogenic_heating(TS82_uranium_params, TS82_thorium_params, TS82_potassium_params, 'Turcotte and Schubert [1984] ')

# Bulk Silicate Earth Models:
CI_uranium_params = {'heat_release_235': H_Uranium235,
                       'heat_release_238': H_Uranium238,
                       'half_life_235': T_Uranium235,
                       'half_life_238': T_Uranium238,
                     'partition_coefficient': D_Wanke_Dreibus_Mars,
                     'bulk_concentration': C_Uranium_CI
}

CI_potassium_params = {'heat_release': H_Potasium40,
                       'half_life': T_Potasium40,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Potassium_CI
}

CI_thorium_params = {'heat_release': H_Thorium,
                     'half_life': T_Thorium232,
                     'partition_coefficient': D_Wanke_Dreibus_Mars,
                     'bulk_concentration': C_Thorium_CI
}
CI = radiogenic_heating(CI_uranium_params, CI_thorium_params, CI_potassium_params,'"CI Carbonaceous Chondrites" [1995]')

# Bulk Silicate Earth Models:
BSE_uranium_params = {'heat_release_235': H_Uranium235,
                      'heat_release_238': H_Uranium238,
                       'half_life_235': T_Uranium235,
                       'half_life_238': T_Uranium238,
                      'partition_coefficient': D_Wanke_Dreibus_Mars,
                      'bulk_concentration': C_Uranium_BSE
}

BSE_potassium_params = {'heat_release': H_Potasium40,
                        'half_life': T_Potasium40,
                        'partition_coefficient': D_Wanke_Dreibus_Mars,
                        'bulk_concentration': C_Potassium_BSE
}

BSE_thorium_params = {'heat_release': H_Thorium,
                      'half_life': T_Thorium232,
                      'partition_coefficient': D_Wanke_Dreibus_Mars,
                      'bulk_concentration': C_Thorium_BSE
}
BSE = radiogenic_heating(BSE_uranium_params, BSE_thorium_params, BSE_potassium_params, '"Bulk Silicate Earth" [1995]')

# Wanke and Dreibus Models:
WD94_uranium_params = {'heat_release_235': H_Uranium235,
                       'heat_release_238': H_Uranium238,
                       'half_life_235': T_Uranium235,
                       'half_life_238': T_Uranium238,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Uranium_Wanke
}

WD94_potassium_params = {'heat_release': H_Potasium40,
                         'half_life': T_Potasium40,
                         'partition_coefficient': D_Wanke_Dreibus_Mars,
                         'bulk_concentration': C_Potassium_Wanke
}

WD94_thorium_params = {'heat_release': H_Thorium,
                       'half_life': T_Thorium232,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Thorium_Wanke
}

WD94 = radiogenic_heating(WD94_uranium_params, WD94_thorium_params, WD94_potassium_params, 'Wanke and Dreibus [1994]')

# Lodders and Fegley Models:
LF97_uranium_params = {'heat_release_235': H_Uranium235,
                       'heat_release_238': H_Uranium238,
                       'half_life_235': T_Uranium235,
                       'half_life_238': T_Uranium238,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Uranium_Lodders
}

LF97_potassium_params = {'heat_release': H_Potasium40,
                         'half_life': T_Potasium40,
                         'partition_coefficient': D_Wanke_Dreibus_Mars,
                         'bulk_concentration': C_Potassium_Lodders
}

LF97_thorium_params = {'heat_release': H_Thorium,
                       'half_life': T_Thorium232,
                       'partition_coefficient': D_Wanke_Dreibus_Mars,
                       'bulk_concentration': C_Thorium_Lodders
}
LF97 = radiogenic_heating(LF97_uranium_params, LF97_thorium_params, LF97_potassium_params, 'Lodders and Fegley [1997]')


def schubert_spoon_heating_model(time):
    return 1.7e-7 * np.exp(-1.38e-17 * (time))/3400.


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg

  fig1 = mpimg.imread('hauck_phillips.png')
  plt.imshow(fig1, extent=[0., 4.5,0,6], aspect='auto')
  plt.xlim(0.,4.5)
  plt.ylim(0,6.)
  time = np.linspace(0., Julian_year * 4.5e9, 1000)
  plt.plot(time/(1.e9*Julian_year), schubert_spoon_heating_model(time)/1.e-11, label='Schubert and Spohn')
  models = [LF97, WD94, CI, TS82 ]
  for ii, model in enumerate(models):
      #plt.subplot(1,len(models),ii)
      heat = model.heat_production(time)
      print heat
      print schubert_spoon_heating_model(time)
      plt.plot(time / (1.e9 * Julian_year), heat/1.e-11, label=model.model_name)
  plt.legend()
  plt.xlabel(r'Time [Ga]')
  plt.ylabel(r'Heat Production [W/kg]')
  plt.title('Heat Production Comparison')
  #plt.gca().invert_xaxis()
  plt.show()

