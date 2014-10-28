'''
mercury_cm_model.py
'''

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

def mercury_model(object): 

    def __init__(core_Mfrac,wS,wSi):
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
        self.M_mantle = M_planet*(1. - core_Mfrac)
        self.M_inner = M_planet*core_Mfrac*inner_Mfrac
        self.M_outer = M_planet*core_Mfrac*(1.-inner_Mfrac)

        #mantle minerals
        n_fe_ol = 0.0 # iron content of mantle minerals
        n_fe_opx = 0.0
        ol = olivine(n_fe_ol)
        opx = orthopyroxene(n_fe_opx)

        # fraction of olivine and orthopyroxene in the mantle
        fol = 0.2; fopx = 1. - fol
        rock = burnman.Composite([fol,fopx],[ol,opx])

        # Total fraction of light elements in the core (in wt. %)
        self.wS = wS
        self.wSi = wSi
        self.wFe = 1. - self.wS - self.wSi

    def set_innerCore(inner_Mfrac,T0=[2200.,1550.,1000.]):
        '''
        Create corePlanet instance with a given inner core mass fraction.

        Should T0 be rearranged in cm_planet to only be called at integration time?
        '''

        # Fraction of the cores mass in the inner core
        assert inner_Mfrac >= 0. and inner_Mfrac <= 1.
        self.inner_Mfrac = 0.5

        # Distribution coefficients [Wsolid]/[Wliquid] (Is this correct, the different
        # weight percents dont take echother into account).
        DS = 0. # DS has to be zero for the current burnman (solid) solution model!!!
        DSi = 1.0

        w_outer, w_inner = partition([wS,wSi],[DS,DSi],inner_Mfrac)

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
        liq_w = Liquidus()
        self.liquidus = lambda p: liq_w.T_SP(self.wS_l,p) 

        # build planet!
        self.planet = corePlanet([M_inner,M_outer,M_mantle],[solidFeSi,liquidFeSSi,rock],T0,
                liquidus=liquidus)

    def integrate(n_slices=300,n_iter=5,P0=40.0e9,plot=False,**kwargs)
        '''
        optional:
            n_slices
            n_iter
            P0
            plot
            verbose
        '''
        # # Integrate!
        planet.integrate(n_slices,P0,n_iter=5,plot=False,**kwargs)
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

    def generate_table(inner_Mfracs,**kwargs):

        toPct = lambda x: str(int(x*100.))
        fname='merc_{}_{}_{}'.format(toPct(self.core_Mfrac,self.wS,self.wSi) )

        for f in inner_Mfracs:
            self.set_innerCore(f)

            

