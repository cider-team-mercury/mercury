import numpy as np
# - Mercury Melt Models for parameterzed convection
# Partial Melting in the mantle, melt extraction, and crustal formation have a been
# shown to have a large effect on the thermal evolution of Mars and Mercury.
# These models follow the work of Breuer and Spohn (2003, 2006), but are modified to
# take into account the solidus increase due to mantle depletion. The presence of
# partial melt in the mantle depends on the solidus temperature of its constituents,
# here we assume the first melting mantle component is peridotite and utilize the
# parameterized solidus from Takahashi (1990).
#
# - Citations:
#
# Breuer D., and T. Spohn. "Early plate tectonics versus single-plate tectonics on Mars: Evidence from magnetic field
#   history and crust evolution." Journal of Geophysical Research: Planets (2003).
#
# Breuer, Doris, and Tilman Spohn. "Viscosity of the Martian mantle and its initial temperature: Constraints from crust
#   formation history and the evolution of the magnetic field." Planetary and Space Science  (2006)
#
# Takahashi, Eiichi. "Speculations on the Archean mantle: missing link between komatiite and depleted garnet
#   peridotite." Journal of Geophysical Research: Solid Earth (1990)
#

def peridotite_solidus(pressure_in_pascals):
    '''
    Calculate the solidus temperature of peridotite as a cubic function of pressure,
    where the pressure is given in units of GPa. The hard coded values here reproduce
    the parameterization of Takahashi (1990) as reported by Morschhauser (2011).

    Equation (16) in Morschhauser et al (2011).
    :param pressure:
    :return:
    '''
    zero_pressure_solidus_temperature = 1409.0
    linear_coef = 134.2
    quadratic_coef = -6.581
    cubic_coef = 0.1054
    pressure_in_GPa = pressure_in_pascals * 1.0e-9
    solidus_temperature = zero_pressure_solidus_temperature + linear_coef*pressure_in_GPa + \
                          quadratic_coef*np.power(pressure_in_GPa, 2.) + cubic_coef*np.power(pressure_in_GPa, 3.)
    return solidus_temperature

def peridotite_liquidus(pressure_in_pascals):
    '''
    Calculate the liquidus temperature of peridotite as a cubic function of pressure,
    where the pressure is given in units of GPa. The hard coded values here reproduce
    the parameterization of Takahashi (1990) as reported by Morschhauser (2011).

     Equation (17) in Morschhauser et al (2011).
    :param pressure_in_pascals:
    :return:
    '''
    zero_pressure_liquidus_temperature = 2035.0
    linear_coef = 57.46
    quadratic_coef = -3.487
    cubic_coef = 0.0769
    pressure_in_GPa = pressure_in_pascals * 1.0e-9
    liquidus_temperature = zero_pressure_liquidus_temperature + linear_coef*pressure_in_GPa + \
                          quadratic_coef*np.power(pressure_in_GPa, 2.) + cubic_coef*np.power(pressure_in_GPa, 3.)
    return liquidus_temperature

def mantle_solidus(pressure_in_pascals, crustal_thickness, inner_mantle_radius, outer_mantle_radius):
    '''
    With the extraction of crustal components from the mantle rock, we assume that the
    solidus will increase linearly until a maximum solidus temperature has been reached,
    equation (18) Morschhauser et al (2011).

    This maximum change is bounded by the solidus difference between un-depleted and depleted
    peridotite, e.g. hazburgite, of 150 K.

     A reference crustal thickness is defined such that the solidus increases the maximum value
     after the extraction of 20 percent of the total silicate volume, equation (19) Morschhauser
     et al (2011).

    :param pressure_in_pascals:
    :return:
    '''
    delta_solidus_temperature = 150.0
    peridotite_solidus_temperature = peridotite_solidus(pressure_in_pascals)
    reference_crustal_thickness = (0.2/3.0) * (np.power(outer_mantle_radius, 3.) - np.power(inner_mantle_radius, 3.))\
                                  /np.power(outer_mantle_radius, 2.)
    return peridotite_solidus_temperature + (crustal_thickness/reference_crustal_thickness) * delta_solidus_temperature

#def volumetric_degree_melting():


