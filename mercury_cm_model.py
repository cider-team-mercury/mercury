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

from build_planet_cm import cm_Planet, corePlanet

from core_partition import partition

# Material Properties
from mercury_minerals import *

# Constants
from mercury_reference import *

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
        self.M_planet = M()[0]
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
        self.planet.integrate(n_slices,P0,n_iter=5,plot=False,**kwargs)
        # print self.planet.moment_over_mr2()

        if plot:
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
            plt.xlabel(r"Radius [$km$]")
            plt.ylabel(r"Temperature [$K$]")

            plt.show()

    def generate_table(self,inner_Mfracs,data_dir="tables/",**kwargs):
        '''
        Generate a table with values for a parameterized convection code.

        inner core mass fraction and radius fraction:
        mfrac, rfrac

        radii (m):
        r_icb,r_cmb,r_surf,
        
        Temperatures (K):
        Ti_avg,Tc_avg,Tm_avg,T_center,T_icb,T_cmb,T_surf,
        
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

            try:
                self.set_innerCore(mfrac) 

                self.integrate(verbose=False,**kwargs)

                r_icb = self.planet.boundaries[0]
                r_cmb = self.planet.boundaries[1]
                r_surf = self.planet.boundaries[-1]
                rfrac = r_icb / r_cmb

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

                rho_diff = 0. # placeholder

                C_MR2 = self.planet.moment_over_mr2()
                Cm_C = self.planet.moment_of_inertia_list()[-1] / self.planet.moment_of_inertia()

                row = np.array([mfrac,rfrac,r_icb,r_cmb,r_surf,T_center,T_icb,T_cmb,T_surf,
                    Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,Cm_C,float(at_eutectic)])
                row_list.append(row)

                at_eutectic = not self.liq_w.is_Fe_rich(w_S,
                        self.planet.pressure[self.planet.inner_core()][0] )
                print w_S,at_eutectic
            except:
                print 'Problem encountered, skipping step without adding to table'
            

        array_to_print = np.vstack(row_list)
        print target
        np.savetxt(target, array_to_print,delimiter=',',
                header='mfrac,rfrac,r_icb,r_cmb,r_surf,T_center,T_icb,T_cmb,T_surf,Ti_avg,Tc_avg,Tm_avg,E_g,w_S,w_Si,rho_diff,C_MR2,Cm_C,at_eutectic')
        return array_to_print

if __name__ == "__main__":
    merc = mercury_model(0.9,.05,.05)
    a1 = merc.generate_table(np.linspace(0.,0.9,50))
#     a2 = merc.generate_table(np.linspace(0.,0.9,10),n_iter=10)
#     a3 = merc.generate_table(np.linspace(0.,0.9,10),n_slices=1000)
# 
#     plt.figure()
#     ax = plt.subplot(111)
#     ax.plot(a1[:,10],a1[:,1])
#     ax.plot(a3[:,10],a2[:,1])
#     ax.plot(a4[:,10],a4[:,1])
#     plt.show()

