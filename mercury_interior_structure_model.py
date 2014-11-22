'''
mercury_cm_model.py
'''

import os, sys

import numpy as np
import matplotlib.pyplot as plt

import burnman
import burnman.minerals as minerals
import burnman.composite as composite


# from liquidus_model import Solver as Liquidus
from liquidus_model import Solver_no14 as FeSLiquidusModel

from build_planet import cm_Planet, corePlanet

from core_partition import partition, density_coexist

# Material Properties
from mercury_minerals import *

# Constants
import mercury_reference as ref

# molar masses
mFe = ref.mFe
mSi = ref.mSi
mS = ref.mS

class mercuryModel(corePlanet): 

    def __init__(self,core_Mfrac,wS,wSi,\
            liquidus=FeSLiquidusModel,T0=[2200.,1550.,1000.],**kwargs):
        '''
        Define a model of mercury with a core with a total mass fraction, S content and 
        Si content.
        '''

        # Fraction of the planets mass in the iron core (inner + outer)
#         core_Mfrac = 0.9
        assert core_Mfrac >= 0. and core_Mfrac <= 1.
        assert wS >= 0. and wS <= 1.
        assert wSi >= 0. and wSi <= 1.

        self.core_Mfrac = core_Mfrac

        # Define masses of layers consistent with mercury_reference
        self.M_planet = ref.M[0]
        self.M_mantle = self.M_planet*(1. - self.core_Mfrac)
        self.M_core = self.M_planet*self.core_Mfrac

        # Total fraction of light elements in the core (in wt. %)
        self.wS = wS
        self.wSi = wSi
        self.wFe = 1. - self.wS - self.wSi

        # dummy compositions and masses (can Planet be changed to not require this)
        masses = [ 0., self.M_core, self.M_mantle]
        compositions = [None,None,None]

        # build planet!
        super(mercuryModel,self).__init__(masses, compositions,T0,\
                liquidus=liquidus, **kwargs )

    def set_innerCore(self,inner_Mfrac):
        '''
        Create corePlanet instance with a given inner core mass fraction.

        Should T0 be rearranged in cm_planet to only be called at integration time?
        '''

        # Fraction of the cores mass in the inner core
        assert inner_Mfrac >= 0. and inner_Mfrac <= 1.
        self.inner_Mfrac = inner_Mfrac

        # inner and outer core (this is redundant)
        self.M_inner = self.M_planet*self.core_Mfrac*self.inner_Mfrac
        self.M_outer = self.M_planet*self.core_Mfrac*(1.-self.inner_Mfrac)

        self.masses = [self.M_inner,self.M_outer,self.M_mantle]
        self.update_massBelowBoundary()

        #mantle minerals
        n_fe_ol = ref.n_fe_ol # iron content of mantle minerals
        n_fe_opx = ref.n_fe_opx
        ol = olivine(n_fe_ol)
        opx = orthopyroxene(n_fe_opx)

        # fraction of olivine and orthopyroxene in the mantle
        fol = ref.fol; fopx = ref.fopx
        rock = burnman.Composite([fol,fopx],[ol,opx])

        # Distribution coefficients [Wsolid]/[Wliquid] (Is this correct, the different
        # weight percents dont take echother into account).
        self.DS = ref.DS 
        self.DSi = ref.DSi

        w_outer, w_inner = partition([self.wS,self.wSi],[self.DS,self.DSi],self.inner_Mfrac)

        self.w_l = np.array([ w_outer[0],w_outer[1],1.-w_outer[0]-w_outer[1]] )
        self.x_l = w_to_x(self.w_l)

        assert np.sum(self.w_l) == 1)
        assert np.all(self.w_l >= 0.)
        assert np.sum(self.x_l) == 1)
        assert np.all(self.x_l >= 0.)

        liquidFeSSi = ironSulfideSilicideLiquid(self.x_l[0],self.x_l[1]) # ternary solution

        self.w_s = np.array([ w_inner[0],w_inner[1],1.-w_inner[0]-w_inner[1]] )
        self.x_s = w_to_x(sesf.w_s)

        assert self.x_s[0] == 0. # DS has to be zero for the current burnman solution model!!!
        assert np.sum(self.w_s) == 1)
        assert np.all(self.w_s >= 0.)
        assert np.sum(self.x_s) == 1)
        assert np.all(self.x_s >= 0.)

        solidFeSi = ironSilicideAlloy(self.x_s[1]) # solid solution of Si in Fe

        # set compositions to the burnman minerals of corresponding composition
        self.compositions = [solidFeSi,liquidFeSSi,rock]

        # build planet!
#         self.planet = corePlanet([self.M_inner,self.M_outer,self.M_mantle],
#                 self.materials,T0,liquidus=liquidus)

    def integrate(self,n_slices=500,P0=40.0e9,n_iter=5,**kwargs):
        """
        Iteratively determine the pressure, density temperature and gravity profiles
        for the planet as a function of radius within a planet.

        Parameters
        ----------
        n_slices : number of steps in integrated mass

        P0 : initial guess for central pressure in Pa

        Optional
        ----------
        n_iter : number of iterations (default: 5)

        profile_type : temperature profile type ('adiabatic' or 'isothermal,
            default: 'adiabatic')

        plot : create plot of density, gravity, pressure and temperature as a 
            function of radius (default: False)
        
        verbose : (default: True)
        """
        # # Integrate!
#         self.planet.integrate(n_slices,P0,n_iter=5,**kwargs)
        super(mercuryModel,self).integrate(n_slices,P0,n_iter=n_iter,**kwargs)

    def generate_profiles(self,inner_Mfrac,plot=False,**kwargs):

            self.set_innerCore(inner_Mfrac) 
            self.integrate(verbose=False,**kwargs)

    def show_profiles(self):
            # testing detection of snowing layers
            print self.planet.detect_snow()
            print self.planet.adiabat_steeper()

            outer_core = self.planet.outer_core()
            p_oc = self.planet.pressure[outer_core]
            r_oc = self.planet.radius[outer_core]
            liq_oc = self.planet.liquidus(p_oc)

            if True:
                # Plot 
                plt.subplot(141)
                plt.plot(self.planet.radial_profile()/1.e3, self.planet.density_profile())
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Density [$kg/m^3$]")

                plt.subplot(142)
                plt.plot(self.planet.radial_profile()/1.e3, self.planet.gravity_profile())
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Gravity [$m/s^2$]")

                plt.subplot(143)
                plt.plot(self.planet.radial_profile()/1.e3, self.planet.pressure_profile()/1.e9)
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Pressure [$Pa$]")

                plt.subplot(144)
                plt.plot(self.planet.radial_profile()/1.e3, self.planet.temperature_profile())
                plt.plot(r_oc/1.e3,liq_oc,'r')
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Temperature [$K$]")

                plt.show()



class model_suite(object):
    def __init__(self,inner_Mfracs,**kwargs):
        self.inner_Mfracs = inner_Mfracs
    def get_energetics(self,**kwargs):

        row_list = []
        at_eutectic = False
        for mfrac in inner_Mfracs:

            print mfrac

            if at_eutectic: # stop if eutectic has been reached
                print 'Eutectic encountered'
                break

            self.set_innerCore(mfrac) 

            self.integrate(verbose=False,**kwargs)

            r_icb = self.planet.boundaries[0]
            r_cmb = self.planet.boundaries[1]
            r_surf = self.planet.boundaries[-1]
            rfrac = r_icb / r_cmb

            dEg_cmb = self.planet.Eg_over_r
            Eg = self.planet.gravitational_energy
            Eg_core = self.planet.core_gravitational_energy

            row = np.array([mfrac,r_icb,dEg_cmb,Eg,Eg_core])
                
            row_list.append(row)

        array_to_print = np.vstack(row_list)
        return array_to_print

#     def generate_table(self,inner_Mfracs,data_dir="tables/",test=False,**kwargs):
#         '''
#         Generate a table with values for a parameterized convection code.
# 
#         inner core mass fraction and radius fraction:
#         mfrac, rfrac
# 
#         radii (m):
#         r_slb,r_rib,r_surf,
# 
#         Pressure(Pa):
#         P_slb
#         
#         Temperatures (K):
#         Ti_avg,Tc_avg,Tm_avg,T_center,T_slb,T_rib,T_surf,
#         
#         Gravitational Energy:
#         E_g,
# 
#         Composition of the liquid (wt. %)
#         w_S,w_Si,
# 
#         Density difference of coexisting solid and liquid:
#         rho_diff
# 
#         moment of inertias:
#         C_MR2, Cm_C
#         
#         Whether calculation has reached the eutectic:
#         at_eutectic,
# 
#         Note: The burnman solid solution models dont allow light element concentrations
#         above a certain amount (so the sulfur content especially will exceed this for
#         thin fluid shells).
#         '''
# 
#         assert os.path.isdir(data_dir)
# 
#         toPct = lambda x: str(int(x*100.))
# 
#         fname='merc_{}_{}_{}.csv'.format(toPct(self.core_Mfrac),toPct(self.wS),toPct(self.wSi) )
#         target = os.path.join(data_dir,fname)
#         
#         row_list = []
#         at_eutectic = False
#         for mfrac in inner_Mfracs:
# 
#             print mfrac
# 
#             if at_eutectic: # stop if eutectic has been reached
#                 print 'Eutectic encountered'
#                 break
# 
# #             try:
#             self.set_innerCore(mfrac) 
# 
#             self.integrate(verbose=False,**kwargs)
# 
#             r_icb = self.planet.boundaries[0]
#             r_cmb = self.planet.boundaries[1]
#             r_surf = self.planet.boundaries[-1]
#             rfrac = r_icb / r_cmb
# 
#             P_icb = self.planet.pressure[self.planet.outer_core()][0]
# 
#             T_center = self.planet.temperature[0]
#             T_icb = self.planet.boundary_temperatures[0]
#             T_cmb = self.planet.boundary_temperatures[1]
#             T_surf = self.planet.boundary_temperatures[-1]
# 
#             Ti_avg = np.mean(self.planet.temperature[self.planet.inner_core()] )
#             Tc_avg = np.mean(self.planet.temperature[self.planet.outer_core()] )
#             Tm_avg = np.mean(self.planet.temperature[self.planet.mantle()] )
# 
#             E_g = 0. # placeholder
# 
#             w_S = self.wS_l
#             w_Si = self.wSi_l
# 
#             rho_icb_s, rho_icb_l = density_coexist([self.wS_l,self.wSi_l,self.wFe_l],\
#                     [self.DS,self.DSi],P_icb,T_icb)
#             rho_diff = rho_icb_s - rho_icb_l 
#             print rho_icb_s, rho_icb_l,rho_diff
# 
#             C_MR2 = self.planet.moment_over_mr2()
#             Cm_C = self.planet.moment_of_inertia_list()[-1] / self.planet.moment_of_inertia()
# 
#             at_eutectic = not self.liq_w.is_Fe_rich(self.wS_l,
#                     self.planet.pressure[self.planet.inner_core()][0] )
# 
#             row = np.array([mfrac,rfrac,r_icb,r_cmb,r_surf,P_icb,T_center,T_icb,T_cmb,T_surf,
#                 Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,Cm_C,float(at_eutectic)])
#             row_list.append(row)
# 
#             print w_S,at_eutectic
# #             except:
# #                 print 'Problem encountered, skipping step without adding to table'
# 
#         csv_header = 'Model of mercury with growing inner core:\n'\
#                 + 'M_core/M= '+str(self.M_core/self.M_planet)\
#                 + ', wS= '+str(self.wS)+', wSi= '+str(self.wSi)+'\n'\
#                 + 'mfrac,rfrac,r_slb,r_rib,r_surf,P_slb,T_center,T_slb,T_rib,'\
#                 + 'T_surf,Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,'\
#                 + 'Cm_C,at_eutectic'
# 
#         array_to_print = np.vstack(row_list)
#         print target
#         if not test:
#             np.savetxt(target, array_to_print,delimiter=',',
#                 header=csv_header)
#         return array_to_print

if __name__ == "__main__":
    # .58,.68,.63 (range in masses found in Hauck)
    merc = mercury_model(0.63,.00,.00)

#     a1 = merc.generate_table(np.linspace(0.,.8,8*4+1))

#     a1 = merc.get_energetics(np.linspace(0.,.5,6))

    merc.generate_profiles(0.1)
    merc.show_profiles()
