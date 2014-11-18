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
from liquidus_model import Solver_no14 as Liquidus

from build_planet import cm_Planet, corePlanet

from core_partition import partition, density_coexist

# Material Properties
from mercury_minerals import *

# Constants
import mercury_reference

# molar masses
mFe = 55.845
mSi = 28.0855
mS = 32.066

class mercury_model(object): 

    def __init__(self,core_Mfrac,wS,wSi):
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
        self.M_planet = mercury_reference.M[0]
        self.M_mantle = self.M_planet*(1. - self.core_Mfrac)
        self.M_core = self.M_planet*self.core_Mfrac

        # Total fraction of light elements in the core (in wt. %)
        self.wS = wS
        self.wSi = wSi
        self.wFe = 1. - self.wS - self.wSi

    def set_innerCore(self,inner_Mfrac,T0=[2200.,1550.,1000.]):
        '''
        Create corePlanet instance with a given inner core mass fraction.

        Should T0 be rearranged in cm_planet to only be called at integration time?
        '''

        # Fraction of the cores mass in the inner core
        assert inner_Mfrac >= 0. and inner_Mfrac <= 1.
        self.inner_Mfrac = inner_Mfrac

        # inner and outer core
        self.M_inner = self.M_planet*self.core_Mfrac*self.inner_Mfrac
        self.M_outer = self.M_planet*self.core_Mfrac*(1.-self.inner_Mfrac)

        #mantle minerals
        n_fe_ol = 0.0 # iron content of mantle minerals
        n_fe_opx = 0.0
        ol = olivine(n_fe_ol)
        opx = orthopyroxene(n_fe_opx)

        # fraction of olivine and orthopyroxene in the mantle
        fol = 0.2; fopx = 1. - fol
        rock = burnman.Composite([fol,fopx],[ol,opx])

        # Distribution coefficients [Wsolid]/[Wliquid] (Is this correct, the different
        # weight percents dont take echother into account).
        self.DS = 0. # DS has to be zero for the current burnman (solid) solution model!!!
        self.DSi = 1.0

        w_outer, w_inner = partition([self.wS,self.wSi],[self.DS,self.DSi],self.inner_Mfrac)

        self.wS_l = w_outer[0]; self.wSi_l=w_outer[1];
        self.wFe_l = 1.-self.wS_l-self.wSi_l
        self.xS_l = (self.wS_l/mS) / ( self.wS_l/mS + self.wFe_l/mFe + self.wSi_l/mSi)
        self.xSi_l = (self.wSi_l/mSi) / ( self.wS_l/mS + self.wFe_l/mFe + self.wSi_l/mSi)
        liquidFeSSi = ironSulfideSilicideLiquid(self.xS_l,self.xSi_l) # ternary solution

        self.wS_s = w_inner[0]; self.wSi_s=w_inner[1];
        self.wFe_s = 1.-self.wS_s-self.wSi_s
        self.xS_s = (self.wS_s/mS) / ( self.wS_s/mS + self.wFe_s/mFe + self.wSi_s/mSi)
        self.xSi_s = (self.wSi_s/mSi) / ( self.wS_s/mS + self.wFe_s/mFe + self.wSi_s/mSi)
        assert self.xS_s == 0. # DS has to be zero for the current burnman solution model!!!
        solidFeSi = ironSilicideAlloy(self.xSi_s) # solid solution of Si in Fe


        # find the T(P) liquidus curve for the given wS (is this absurdly slow?)
        # could always refit, or add a function to the liquidus model
        self.liq_w = Liquidus()
        liquidus = lambda p: self.liq_w.T_SP(self.wS_l,p) 
        self.materials = [solidFeSi,liquidFeSSi,rock]

        # build planet!
        self.planet = corePlanet([self.M_inner,self.M_outer,self.M_mantle],
                self.materials,T0,liquidus=liquidus)

    def integrate(self,n_slices=500,n_iter=5,P0=40.0e9,plot=False,**kwargs):
        '''
        optional:
            n_slices
            n_iter
            P0
            plot
            verbose
        '''
        # # Integrate!
        self.planet.integrate(n_slices,P0,n_iter=5,**kwargs)
        # print self.planet.moment_over_mr2()


    def generate_table(self,inner_Mfracs,data_dir="tables/",test=False,**kwargs):
        '''
        Generate a table with values for a parameterized convection code.

        inner core mass fraction and radius fraction:
        mfrac, rfrac

        radii (m):
        r_slb,r_rib,r_surf,

        Pressure(Pa):
        P_slb
        
        Temperatures (K):
        Ti_avg,Tc_avg,Tm_avg,T_center,T_slb,T_rib,T_surf,
        
        Gravitational Energy:
        E_g,

        Composition of the liquid (wt. %)
        w_S,w_Si,

        Density difference of coexisting solid and liquid:
        rho_diff

        moment of inertias:
        C_MR2, Cm_C
        
        Whether calculation has reached the eutectic:
        at_eutectic,

        Note: The burnman solid solution models dont allow light element concentrations
        above a certain amount (so the sulfur content especially will exceed this for
        thin fluid shells).
        '''

        assert os.path.isdir(data_dir)

        toPct = lambda x: str(int(x*100.))

        fname='merc_{}_{}_{}.csv'.format(toPct(self.core_Mfrac),toPct(self.wS),toPct(self.wSi) )
        target = os.path.join(data_dir,fname)
        
        row_list = []
        at_eutectic = False
        for mfrac in inner_Mfracs:

            print mfrac

            if at_eutectic: # stop if eutectic has been reached
                print 'Eutectic encountered'
                break

#             try:
            self.set_innerCore(mfrac) 

            self.integrate(verbose=False,**kwargs)

            r_icb = self.planet.boundaries[0]
            r_cmb = self.planet.boundaries[1]
            r_surf = self.planet.boundaries[-1]
            rfrac = r_icb / r_cmb

            P_icb = self.planet.pressure[self.planet.outer_core()][0]

            T_center = self.planet.temperature[0]
            T_icb = self.planet.boundary_temperatures[0]
            T_cmb = self.planet.boundary_temperatures[1]
            T_surf = self.planet.boundary_temperatures[-1]

            Ti_avg = np.mean(self.planet.temperature[self.planet.inner_core()] )
            Tc_avg = np.mean(self.planet.temperature[self.planet.outer_core()] )
            Tm_avg = np.mean(self.planet.temperature[self.planet.mantle()] )

            E_g = 0. # placeholder

            w_S = self.wS_l
            w_Si = self.wSi_l

            rho_icb_s, rho_icb_l = density_coexist([self.wS_l,self.wSi_l,self.wFe_l],\
                    [self.DS,self.DSi],P_icb,T_icb)
            rho_diff = rho_icb_s - rho_icb_l 
            print rho_icb_s, rho_icb_l,rho_diff

            C_MR2 = self.planet.moment_over_mr2()
            Cm_C = self.planet.moment_of_inertia_list()[-1] / self.planet.moment_of_inertia()

            at_eutectic = not self.liq_w.is_Fe_rich(self.wS_l,
                    self.planet.pressure[self.planet.inner_core()][0] )

            row = np.array([mfrac,rfrac,r_icb,r_cmb,r_surf,P_icb,T_center,T_icb,T_cmb,T_surf,
                Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,Cm_C,float(at_eutectic)])
            row_list.append(row)

            print w_S,at_eutectic
#             except:
#                 print 'Problem encountered, skipping step without adding to table'

        csv_header = 'Model of mercury with growing inner core:\n'\
                + 'M_core/M= '+str(self.M_core/self.M_planet)\
                + ', wS= '+str(self.wS)+', wSi= '+str(self.wSi)+'\n'\
                + 'mfrac,rfrac,r_slb,r_rib,r_surf,P_slb,T_center,T_slb,T_rib,'\
                + 'T_surf,Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,'\
                + 'Cm_C,at_eutectic'

        array_to_print = np.vstack(row_list)
        print target
        if not test:
            np.savetxt(target, array_to_print,delimiter=',',
                header=csv_header)
        return array_to_print

    def get_energetics(self,inner_Mfracs,**kwargs):

        row_list = []
        at_eutectic = False
        for mfrac in inner_Mfracs:

            print mfrac

            if at_eutectic: # stop if eutectic has been reached
                print 'Eutectic encountered'
                break

#             try:
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


if __name__ == "__main__":
    # .58,.68,.63 (range in masses found in Hauck)
    merc = mercury_model(0.63,.00,.00)

#     a1 = merc.generate_table(np.linspace(0.,.8,8*4+1))

#     a1 = merc.get_energetics(np.linspace(0.,.5,6))

    merc.generate_profiles(0.5)
    merc.show_profiles()
