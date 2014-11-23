'''
mercury_cm_model.py
'''

import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import burnman
import burnman.minerals as minerals
import burnman.composite as composite

from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative

# from liquidus_model import Solver as Liquidus
from liquidus_model import Solver_no14 as FeSLiquidusModel
# from liquidus_model import Solver as FeSLiquidusModel

from build_planet import cm_Planet, corePlanet

from core_partition import partition, density_coexist,w_to_x,x_to_w

# Material Properties
from mercury_minerals import olivine,orthopyroxene,\
        ironSilicideAlloy,ironSulfideSilicideLiquid

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

        assert np.sum(self.w_l) == 1.
        assert np.all(self.w_l >= 0.)
        assert np.sum(self.x_l) == 1.
        assert np.all(self.x_l >= 0.)

        liquidFeSSi = ironSulfideSilicideLiquid(self.x_l[0],self.x_l[1]) # ternary solution

        self.w_s = np.array([ w_inner[0],w_inner[1],1.-w_inner[0]-w_inner[1]] )
        self.x_s = w_to_x(self.w_s)

        assert self.x_s[0] == 0. # DS has to be zero for the current burnman solution model!!!
        assert np.sum(self.w_s) == 1.
        assert np.all(self.w_s >= 0.)
        assert np.sum(self.x_s) == 1.
        assert np.all(self.x_s >= 0.)

        solidFeSi = ironSilicideAlloy(self.x_s[1]) # solid solution of Si in Fe

        # set compositions to the burnman minerals of corresponding composition
        self.set_compositions([solidFeSi,liquidFeSSi,rock])

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
            print self.detect_snow()
            print self.adiabat_steeper()

            outer_core = self.outer_core()
            core = self.core()
            p_c = self.pressure[core]
            r_c = self.radius[core]
            liq_c = self.liquidus(p_c)

            if True:
                # Plot 
                plt.subplot(141)
                plt.plot(self.radial_profile()/1.e3, self.density_profile())
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Density [$kg/m^3$]")

                plt.subplot(142)
                plt.plot(self.radial_profile()/1.e3, self.gravity_profile())
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Gravity [$m/s^2$]")

                plt.subplot(143)
                plt.plot(self.radial_profile()/1.e3, self.pressure_profile()/1.e9)
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Pressure [$Pa$]")

                plt.subplot(144)
                plt.plot(self.radial_profile()/1.e3, self.temperature_profile())
                plt.plot(r_c/1.e3,liq_c,'r')
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Temperature [$K$]")

                plt.show()

class model_suite(object):
    def __init__(self,planet,inner_Mfracs,**kwargs):
        self.inner_Mfracs = inner_Mfracs
        self.planet = planet
    def get_energetics(self,**kwargs):

        row_list = []
        at_eutectic = False
        self.labels = ['m_frac','r_frac','r_icb','r_cmb','r_surf','T_icb','T_cmb',\
                'T_avg_ic','T_av_oc','Eg_r','L_r','Eg_m','L_m','Cp_ic','Cp_oc']
        for mfrac in self.inner_Mfracs:

            print 'Core mass fraction:', mfrac

            if at_eutectic: # stop if eutectic has been reached
                print 'Eutectic encountered'
                break

#             try:
            self.planet.set_innerCore(mfrac) 

            self.planet.integrate(verbose=False,**kwargs)


            T_icb = self.planet.boundary_temperatures[0]
            T_cmb = self.planet.boundary_temperatures[1]

            m_ic = self.planet.masses[0]
            m_oc = self.planet.masses[1]

            r_icb = self.planet.boundaries[0]
            r_cmb = self.planet.boundaries[1]
            r_surf = self.planet.boundaries[-1]
            rfrac = r_icb / r_cmb

            Cp_avg = self.planet.average_heat_capacity()
            T_avg = self.planet.average_temperature()

            Eg_m = self.planet.specific_gravitational_energy()
            Eg_r = self.planet.gravitational_energy_over_r()

            L_r = self.planet.latent_heat_over_r()
            L_m = self.planet.specific_latent_heat()


            row = np.array([mfrac,rfrac,r_icb,r_cmb,r_surf,T_icb,T_cmb,\
                    T_avg[0],T_avg[1],Eg_r,L_r,Eg_m,L_m,Cp_avg[0],Cp_avg[1]])
            print row
                
            row_list.append(row)
#             except:
#                 print 'Problem encountered, skipping step without adding data'

        print row_list
        self.data = pd.DataFrame(row_list)
        self.data.columns = self.labels
        self.T_max = self.data.T_icb.max()
        self.T_min = self.data.T_icb.min()
        
    def saveData(self,file_name):
        self.data.save(file_name)
    def loadData(self,file_name):
        pd.load(file_name)
    def printData(self):
        print self.data
    def func_of_Ticb(self,label):
        y = self.data[label]
        x = self.data.T_icb

        return UnivariateSpline(x,y)

if __name__ == "__main__":
    # .58,.68,.63 (range in masses found in Hauck)
    merc = mercuryModel(0.63,.00,.00)

#     merc.generate_profiles(0.5)
#     merc.show_profiles()

    model1 = model_suite(merc,[0.1,0.2,0.3,0.4,0.5])
    model1.get_energetics()
    model1.printData()
    model1.saveData('tables/energetics_63_00_00.dat')
    model1.loadData('tables/energetics_63_00_00.dat')
    model1.printData()

    r_func = model1.func_of_Ticb('r_icb')
    Eg_r_func = model1.func_of_Ticb('Eg_r')
    L_r_func = model1.func_of_Ticb('L_r')
    Eg_m_func = model1.func_of_Ticb('Eg_m')
    L_m_func = model1.func_of_Ticb('L_m')

    t = np.linspace(model1.T_min,model1.T_max,50)

    f1 = plt.figure()
    ax1= plt.subplot(111)
    ax1.plot(t,r_func(t))

    # check units on these
    f2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.plot(t,Eg_r_func(t))
    ax2.plot(t,L_r_func(t))

    f3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.plot(t,Eg_m_func(t))
    ax3.plot(t,L_m_func(t))

    plt.show()

