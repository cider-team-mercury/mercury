# - Mantle Energry Balance Model for paramterized convection. This mantle layer follows 
#   those implemented in the Thermal Evolution studies of Morschhauser et al 2011 and 
#   Grott et al 2011. 
#   
# - Currently implements temperature dependent viscosity mantle convection in the 
#   Stagnat Lid regeime, in addition to accounting for crustal growth, and partial melting. 
#   
# - Citations:
'''
Morschhauser, A., Grott, M., & Breuer, D. (2011). Crustal recycling, mantle dehydration,
and the thermal evolution of Mars. Icarus, 212(2), 541-558.

Grott, M., Breuer, D., & Laneuville, M. (2011). Thermo-chemical evolution and global 
contraction of Mercury. Earth and Planetary Science Letters, 307(1), 135-146.

Grasset, O., & Parmentier, E. M. (1998). Thermal convection in a volumetrically heated,
infinite Prandtl number fluid with strongly temperature-dependent viscosity: Implications
for planetary thermal evolution. Journal of Geophysical Research: Solid Earth (1978-2012),
103(B8), 18171-18181.
'''


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
from scipy.misc import derivative
import planetary_energetics
from mercury_parameters import mantle_params

class mantle_layer(planetary_energetcs.Layer):
    '''
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
    '''
    def __init__(self,inner_radius,outer_radius, params=mantle_params):
        planetary_energetics.Layer.__init__(self,inner_radius,outer_radius,params)
        '''
        Default Values for Mercury are loaded from the file mercury_paramaters, which are 
        the same as in Grott et al 2011 which implements the model of Morchhauser et al 2011
        for Mercury rather than Mars.
        '''

    def effective_heat_capacity(self):
        '''
        All terms multiplying the change in temperature with time(dT_dt).
        
        For now we have ignored the effect of mantle melting, to include the effect of melting
        simply multiply by (1+St), where St is the Stefan number. See the LHS of equation (1)
        in Morschhauser et al 2011. The definition of the Stefan number is given in equation (2).
        '''
        return self.params['density']*self.params['heat_capacity']*self.params['epsilon']*self.volume

    def upper_boundary_layer_thickness(self, stagnant_lid_thickness):
        '''
        Thickness of the boundary layer at the top of the convecting mantle, equation (12) in
        Morschhauser et al 2011. 

        The thickness of this boundary is calcualted using the Nusselt-Rayleigh relation and 
        critical boundary layer theory, see Turcotte and Schubert 2002.
        '''
        return (self.thickness - stagnant_lid_thickness)*(self.params['Ra_crit']/Ra)^(1./3.)

    def lower_boundary_layer_thickness(self, stagnant_lid_thickness):
        '''
        Thickness of the boundary layer at the base of the convecting mantle, equation (13) in
        Morschhauser et al 2011. 
        '''
        return  (self.thickness - stagnant_lid_thickness)*(self.params['Ra_crit']/Ra)^(1./3.)
    
    def upper_heat_flux(self, T_upper_mantle, stagnant_lid_thickness):
        '''
        Heat flow out of the mantle, equation (10) in Morschhhauser et al 2011.

        The temperature at the base of the Stagnant lid is found using the formulation
        of Grasset and Parmentier et al 1998, adapted to spherical geometry, equation (8)
        in Morschhauser et al 2011. 
        '''
        
        T_base_stagnant_lid = self.calculate_temperature_base_stagnant_lid(T_upper_mantle)  
        Ra = self.calculate_rayleigh_number(T_upper_mantle, stagnant_lid_thickness)
        upper_boundary_layer_thickness = self.upper_boundary_layer_thickness( T_upper_mantle, stagnant_lid_thickness)
        return "" 

    def rayleigh_number(self, T_upper_mantle, stagnant_lid_thickness):
        pass

    def calculate_temperature_base_stagnant_lid(self, T_upper_mantle):
        '''
        Temperature at the base of the Stagnant Lid, equation (8) in Morschhauser et al 2011. 
        
        In previous numerical models and experiments, this temperature was found to be the temp.
        in which the viscosity had grown by an order of magnitude with respect to the convecting mantle.
        See Grasset and Parmentier 1998.
        '''
        Theta = self.params['empirical_spherical_stagnant_lid_param']
        T_drop = Theta*self.params['gas_constant']*np.power(T_upper_mantle,2.)/self.params['activation_energy']
        return T_upper_mantle - T_drop

    def calculate_temperature_base_mantle(self, T_upper_mantle, stagnant_lid_thickness):
        '''
        Temperature at the base of the Mantle, equation (9) in Morschhauser et al 2011. 

        The temperature at the base of the mantle is found by determining the adaibatic increase in the mantle.
        However, to do this correctly we need to know the thickness of the Stagnant Lid, and the thickness of
        the boundary layers at the base of the mantle and the boundary layer at the base of the Stagnant Lid/top
        of the convecting mantle.
        '''
        upper_boundary_layer_thickness = self.upper_mantle_thickness(T_upper_mantle, stagnant_lid_thickness)
        lower_boundary_layer_thickness = self.lower_mantle_thickness(T_upper_mantle, stagnant_lid_thickness)
        thickness_of_convecting_mantle = self.thickness-stagnant_lid_thickness-upper_boundary_layer_thickness\
                                        -lower_boundary_layer_thickness
        alpha = self.params['thermal_expansivity']
        g = self.params['surface_gravity']
        




