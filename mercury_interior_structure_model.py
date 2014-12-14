'''
mercury_cm_model.py
'''

import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.testing import assert_approx_equal 

import burnman
import burnman.minerals as minerals
import burnman.composite as composite

from scipy.interpolate import UnivariateSpline

# from liquidus_model import Solver as Liquidus
# from liquidus_model import Solver_no14 as FeSLiquidusModel
# from liquidus_model import Solver as FeSLiquidusModel
from liquidus_model import Dumberry_liquidus as FeSLiquidusModel

from build_planet import cm_Planet, corePlanet

from core_partition import partition, density_coexist,w_to_x,x_to_w,\
        coeff_comp_expansivity

# Material Properties
from mercury_minerals import olivine,orthopyroxene,\
        ironSilicideAlloy,ironSulfideSilicideLiquid,ironSulfideSilicideLiquid_highExpansivity

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

        # Low Expansivity liquid iron
        materials = [ironSilicideAlloy,ironSulfideSilicideLiquid,\
                [olivine,orthopyroxene]]

        # High Expansivity liquid iron
#         materials = [ironSilicideAlloy,ironSulfideSilicideLiquid_highExpansivity,\
#                 [olivine,orthopyroxene]]

        # build planet!
        super(mercuryModel,self).__init__(masses, compositions,T0,\
                liquidus=liquidus,materials=materials, **kwargs )

    def set_compositions(self,**kwargs):
        '''
        Set compositions (burnman.Material instances) corresponding to
        x_l, x_s and mantle mineral parameters defined in mercury_minerals.
        '''

        #mantle minerals
        n_fe_ol = ref.n_fe_ol # iron content of mantle minerals
        n_fe_opx = ref.n_fe_opx
        ol = self.materials[2][0](n_fe_ol)
        opx = self.materials[2][1](n_fe_opx)

        # fraction of olivine and orthopyroxene in the mantle
        fol = ref.fol; fopx = ref.fopx
        rock = burnman.Composite([fol,fopx],[ol,opx])

        # liquid outer core
        liquidFeSSi = self.materials[1](self.x_l[0],self.x_l[1]) # ternary solution

        # solid inner core
        solidFeSi = self.materials[0](self.x_s[1]) # solid solution of Si in Fe

        # set materials for each layer
#         self.materials = [ironSilicideAlloy,ironSulfideSilicideLiquid,\
#                 [olivine,orthopyroxene]]
        
        # set self.compositions and set methods
        super(mercuryModel,self).set_compositions([solidFeSi,liquidFeSSi,rock])

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

        # Distribution coefficients [Wsolid]/[Wliquid] (Is this correct, the different
        # weight percents dont take echother into account).
        self.DS = ref.DS 
        self.DSi = ref.DSi

        w_outer, w_inner = partition([self.wS,self.wSi],[self.DS,self.DSi],self.inner_Mfrac)

        self.w_l = np.array([ w_outer[0],w_outer[1],1.-w_outer[0]-w_outer[1]] )
        self.x_l = w_to_x(self.w_l)

        assert_approx_equal( np.sum(self.w_l) , 1.)
        assert np.all(self.w_l >= 0.)
        assert_approx_equal( np.sum(self.x_l) , 1.)
        assert np.all(self.x_l >= 0.)


        self.w_s = np.array([ w_inner[0],w_inner[1],1.-w_inner[0]-w_inner[1]] )
        self.x_s = w_to_x(self.w_s)

        assert self.x_s[0] == 0. # DS has to be zero for the current burnman solution model!!!
        assert_approx_equal(np.sum(self.w_s) , 1.)
        assert np.all(self.w_s >= 0.)
        assert_approx_equal(np.sum(self.x_s) , 1.)
        assert np.all(self.x_s >= 0.)

        # set compositions to the burnman minerals of corresponding composition
        self.set_compositions()

        # build planet!
#         self.planet = corePlanet([self.M_inner,self.M_outer,self.M_mantle],
#                 self.materials,T0,liquidus=liquidus)

    def integrate(self,n_slices=2000,P0=40.0e9,n_iter=5,**kwargs):
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

    def show_profiles(self,fname=None):
            # testing detection of snowing layers
#             print self.detect_snow()
#             print self.adiabat_steeper()

            outer_core = self.outer_core()
            core = self.core()
            p_c = self.pressure[core]
            r_c = self.radius[core]
            liq_c = self.liquidus(p_c)

            if True:
                # Plot 
                plt.subplot(221)
                plt.plot(self.radial_profile()/1.e3, self.density_profile(),lw=2)
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Density [$kg/m^3$]")

                plt.subplot(222)
                plt.plot(self.radial_profile()/1.e3, self.gravity_profile(),lw=2)
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Gravity [$m/s^2$]")

                plt.subplot(223)
                plt.plot(self.radial_profile()/1.e3,\
                        self.pressure_profile()/1.e9,lw=2)
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Pressure [$Pa$]")

                plt.subplot(224)
                plt.plot(self.radial_profile()/1.e3, self.temperature_profile(),lw=2)
#                 plt.plot(r_c/1.e3,liq_c,'r')
                plt.xlabel(r"Radius [$km$]")
                plt.ylabel(r"Temperature [$K$]")

                if isinstance(fname,str):
                    plt.savefig(fname)

                plt.show()

class model_suite(object):
    def __init__(self,planet,inner_Mfracs,**kwargs):
        '''
        Definte a model of a planet to determine energetics of core growth.
        -----------------------------------------------------------------------
        args:

            planet: corePlanet or mercuryModel object

            inner_Mfracs : array of mass fractions of inner core (0,1)

        -----------------------------------------------------------------------
        '''
        self.inner_Mfracs = inner_Mfracs
        self.planet = planet
    def get_energetics(self,**kwargs):
        '''
        Tabulates quantities for use with parameterized convection code:

        -----------------------------------------------------------------------
        Quantaties:

            'm_frac' : Fraction of the core mass in the solid inner core.
            'r_frac' : Fraction of the core radius in the solid inner core.
            'r_icb', 'r_cmb', 'r_surf' : boundary radii (m)
            'T_cen','T_icb', 'T_cmb' : boundary temperature (K)
            'T_avg_ic', 'T_av_oc' : mass averaged temperature (K)
            'Eg_r': Gravitational energy release per change in inner core 
                    radius (J/m)
            'L_r' : Latent heat release per change in inner core radius (J/m)
            'Eg_m' : Gravitational energy release per change inner core 
                    mass (J/kg)
            'L_m' : Latent heat release per change in inner core mass (J/kg)
            'Cp_ic','Cp_oc' : Average specific heat capcity for each 
                    layer (J/K/kg)
            'CpT_avg_ic','CpT_avg_oc': Average Cp*T for each layer (J/K)
            'm_ic','m_oc': mass of inner and outer core (kg)
            'c_r': Mass of light element released per change in core radius (kg/m)
            'w_bulk': Wt % light element of the bulk core
            'w_l': Wt % light element in the outer core
            'w_s': Wt % light element in the inner core
            'rho_cen': density at the center of the planet (kg/m^3)
                        (at current T_cen)
            'P_cen','P_icb','P_cmb': pressure at boundaries (Pa)
            'rho_liq_0': Density of the liquid alloy at P=0 (kg/m^3) 
                        (at average outer core temperature)
            'K_liq_0': Bulk modulus (Kt) of the core at P=0 (Pa)
                        (at average outer core temperature)
            'alpha_t': Coefficient of thermal expansivity (of liquid at icb) (1/K)
            'alpha_c': Coefficient of compositional expansivity (of liquid at icb)
                    (1/K).
        '''

        row_list = []
        at_eutectic = False
        self.labels = ['m_frac','r_frac','r_icb','r_cmb','r_surf','T_cen','T_icb',\
                'T_cmb','T_avg_ic','T_av_oc','Eg_r','L_r','Eg_m','L_m','Cp_ic',\
                'Cp_oc','CpT_avg_ic','CpT_avg_oc','m_ic','m_oc','c_r','w_bulk','w_l',\
                'w_s','P_cen','P_icb','P_cmb','rho_cen','rho_liq_0','K_liq_0',\
                'alpha_t','alpha_c']
        for mfrac in self.inner_Mfracs:

            print 'Core mass fraction:', mfrac

            if at_eutectic:
                print "Liquidus encountered. Terminating calculation"
                break

            try:
                self.planet.set_innerCore(mfrac) 
            except:
                print "Problem encountered with inner core setting, probably "\
                        +"outside the range of valid material composition"
                break

            try:
                self.planet.integrate(verbose=False,**kwargs)

                T_cen = self.planet.temperature[0]
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
                CpT_avg = self.planet.specific_thermal_energy()

                Eg_m = self.planet.specific_gravitational_energy()
                Eg_r = self.planet.gravitational_energy_over_r()

                L_r = self.planet.latent_heat_over_r()
                L_m = self.planet.specific_latent_heat()

                c_r = self.planet.light_element_release_over_r()

                w_bulk = self.planet.wS + self.planet.wSi
                w_l = np.sum(self.planet.w_l[:-1])
                w_s = np.sum(self.planet.w_s[:-1])

                P_cen = self.planet.pressure[0]
                P_icb = self.planet.pressure[self.planet.icb()]
                P_cmb = self.planet.pressure[self.planet.cmb()]

                rho_cen = self.planet.density[0]

                liq = self.planet.compositions[1]
                liq.set_method('slb3')
                liq.set_state(0.,T_avg[1])

                rho_liq_0 = liq.density()
                K_liq_0 = liq.K_T

                liq.set_state(P_icb,T_icb)
                alpha_t = liq.alpha

                mat = self.planet.materials[1]
                alpha_c = coeff_comp_expansivity(mat,self.planet.w_l,P_icb,T_icb)

                row = np.array([mfrac,rfrac,r_icb,r_cmb,r_surf,T_cen,T_icb,T_cmb,\
                        T_avg[0],T_avg[1],Eg_r,L_r,Eg_m,L_m,Cp_avg[0],Cp_avg[1],\
                        CpT_avg[0],CpT_avg[1],m_ic,m_oc,c_r,w_bulk,w_l,w_s,\
                        P_cen,P_icb,P_cmb,rho_cen,rho_liq_0,K_liq_0,\
                        alpha_t,alpha_c])

                at_eutectic = not self.planet.liquidus_model.is_Fe_rich(self.planet.w_l[0],\
                       self.planet.pressure[self.planet.icb()])

                row_list.append(row)
            except:
                "Problem encountered during integration. Skipping step."

#         print row_list
        self.data = pd.DataFrame(row_list)
        self.data.columns = self.labels
#         self.T_max = self.data.T_icb.max()
#         self.T_min = self.data.T_icb.min()
        
    def saveData(self,file_name):
        '''
        Save tabulated data for a set of runs to a file
        '''
        self.data.save(file_name)
    def loadData(self,file_name):
        '''
        Load tabulated data for a set of runs from a file
        '''
        self.data = pd.load(file_name)
    def printData(self,quantaties=None):
        '''
        Print data with optional specification of quantity names. If not specified
        all saved quantities are output.
        '''
        if quantaties is None:
            print self.data
        else:
            print self.data[quantaties]
    def func_of_Ticb(self,label,**kwargs):
        '''
        Fit a function as of a given quantity w. r. t. the inner core
        boundary temperature.
        '''
        y = self.data[label][::-1] 
        x = self.data.T_icb[::-1] # Note x must be increasing for Univariatespline

        return UnivariateSpline(x,y,**kwargs)
    
    def func_of_Tcmb(self,label,**kwargs):
        '''
        Fit a function as of a given quantity w. r. t. the core mantle
        boundary temperature.
        '''
        y = self.data[label][::-1] 
        x = self.data.T_cmb[::-1] # Note x must be increasing for Univariatespline

        return UnivariateSpline(x,y,**kwargs)

    def func_of_data(self,xlabel,ylabel,**kwargs):
        '''
        Fit a spline function of one quantity w. r. t. another quantity 
        for the growth of the inner core
        '''
        if self.data[xlabel][0] > np.array(self.data[xlabel])[-1]:
            y = self.data[ylabel][::-1] 
            x = self.data[xlabel][::-1] # Note x must be increasing for Univariatespline
        else:
            y = self.data[ylabel]
            x = self.data[xlabel]

        return UnivariateSpline(x,y,**kwargs)

    def func_of_ricb(self,label,**kwargs):
        return self.func_of_data('r_icb',label,**kwargs)

    def thermal_energy_change(self,**kwargs):
        m_ic_func = self.func_of_Tcmb('m_ic')
        m_oc_func = self.func_of_Tcmb('m_oc')
        CpT_avg_ic_func = self.func_of_Tcmb('CpT_avg_ic',**kwargs) # J / kg
        CpT_avg_oc_func = self.func_of_Tcmb('CpT_avg_oc',**kwargs)
        Eth_ic = lambda t : CpT_avg_ic_func.derivative()(t) * m_ic_func(t) # J / K
        Eth_oc = lambda t : CpT_avg_oc_func.derivative()(t) * m_oc_func(t)
        return Eth_ic, Eth_oc

    def get_effective_core_heat_capacity(self):
        m_func = model1.func_of_Tcmb('m_ic',s=2.e42)
        dm_dT_cmb = m_func.derivative() # kg / K
        Eg_m_func = model1.func_of_Tcmb('Eg_m',s=0)
        L_m_func = model1.func_of_Tcmb('L_m',s=0)
        dEth_ic, dEth_oc = model1.thermal_energy_change(s=1.e8)
        thermal_energy_change = lambda t_cmb: dEth_ic(t_cmb) + dEth_oc(t_cmb)
        gravitational_energy_release = lambda t_cmb: Eg_m_func(t_cmb)*dm_dT_cmb(t_cmb)
        latent_heat = lambda t_cmb: L_m_func(t_cmb)*dm_dT_cmb(t_cmb)
        total = lambda t_cmb: thermal_energy_change(t_cmb) - gravitational_energy_release(t_cmb) - latent_heat(t_cmb)
        return thermal_energy_change, gravitational_energy_release, latent_heat, total

# Define a mercury model with a given total core mass and
# .58,.68,.63 (range in masses found in Hauck)
merc = mercuryModel(0.63,.06,.00)
mfracs = np.linspace(0.,0.8,41)
model1 = model_suite(merc,mfracs)
model1.loadData('../tables/highres2_63_06_00.dat')

if __name__ == "__main__":
    fig_size = [1000/72.27 ,800/72.27]
    params = {'backend': 'ps', 'axes.labelsize': 28, 'text.fontsize': 28,
            'legend.fontsize': 28,
              'xtick.labelsize': 22, 'ytick.labelsize': 22, 
              'xtick.major.size': 10,'ytick.major.size': 10,
              'xtick.minor.size': 6,'ytick.minor.size': 6,
              'xtick.major.width': 2,'ytick.major.width': 2,
              'xtick.minor.width': 2,'ytick.minor.width': 2,
              'axes.linewidth': 2, 'xaxis.labelpad' : 50,
              'text.usetex': False, 'figure.figsize': fig_size,
              'figure.subplot.bottom': 0.100,'figure.subplot.top': 0.980,'figure.subplot.left': 0.130,'figure.subplot.right': 0.950}
#    plt.rcParams.update(params)
    # use latex
    plt.rc('text', usetex=False)
    plt.rc('font',family='sans-serif')

    # Generate profiles

    # Define a mercury model with a given total core mass and 
    # .58,.68,.63 (range in masses found in Hauck)
    merc = mercuryModel(0.63,.06,.00)        

    # Tabulate and save energetics for a suite of models with a growing core.
#     mfracs = np.hstack((np.linspace(0.,0.1,11),np.linspace(0.15,0.8,14)) )
    mfracs = np.linspace(0.,0.8,41)
    model1 = model_suite(merc,mfracs)
#     model1.get_energetics()
#     model1.printData()
#     model1.saveData('tables/highres2_63_06_00.dat')

    # Load results from a saved model suite.
    model1.loadData('tables/highres2_63_06_00.dat')
#     model1.loadData('tables/energetics_63_09_00.dat')
    model1.printData()


#     ### Test 1: Look at profiles and determine whether snow predicted
#     merc.generate_profiles(0.5)
#     merc.show_profiles()

#     ### Test 2: Fit functions of Tcmb and plot energetic quantities

    m_func = model1.func_of_Tcmb('m_ic',s=2.e42)
    dm_dT_cmb = m_func.derivative() # kg / K

    #wrap to get rid of unrealistic negative quantities
    Eg_m_func = model1.func_of_Tcmb('Eg_m',s=0)
    L_m_func = model1.func_of_Tcmb('L_m',s=0)

    dEth_ic, dEth_oc = model1.thermal_energy_change(s=1.e8)

    t_cmb = np.linspace(model1.data.T_cmb.min(),model1.data.T_cmb.max(),100)
    
    # functions for parameterized convection
    thermal_energy_change = lambda t_cmb: dEth_ic(t_cmb) + dEth_oc(t_cmb)
    gravitational_energy_release = lambda t_cmb: Eg_m_func(t_cmb)*dm_dT_cmb(t_cmb)
    latent_heat = lambda t_cmb: L_m_func(t_cmb)*dm_dT_cmb(t_cmb)

    f1 = plt.figure()
    ax1= plt.subplot(111)
    ax1.plot(t_cmb,m_func(t_cmb))
    ax1.plot(model1.data.T_cmb,model1.data.m_ic,'bo')
    ax1.set_xlabel('T_cmb (K)')
    ax1.set_ylabel('R_icb (m)')

    f2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.plot(t_cmb,Eg_m_func(t_cmb))
    ax2.plot(t_cmb,L_m_func(t_cmb))
    ax2.plot(model1.data.T_cmb,model1.data.Eg_m,'bo')
    ax2.plot(model1.data.T_cmb,model1.data.L_m,'go')
    ax2.set_xlabel('T_cmb (K)')
    ax2.set_ylabel('dE/dM_ic (J/kg)')

    f3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.axvline(1498,linestyle='--',color='k',lw=2)
    ax3.axvline(1704,linestyle='-.',lw=2,color='k')
    ax3.plot(t_cmb,-Eg_m_func(t_cmb)*dm_dT_cmb(t_cmb)/1.e26,label=r'$dE_g/dT_{\rm cmb}$',lw=2)
    ax3.plot(t_cmb,-L_m_func(t_cmb)*dm_dT_cmb(t_cmb)/1.e26,label=r'$dE_L/dT_{\rm cmb}$',lw=2)
    ax3.plot(t_cmb,dEth_ic(t_cmb)/1.e26,label=r'$dE_{\rm th,oc}/dT_{\rm cmb}$',lw=2)
    ax3.plot(t_cmb,dEth_oc(t_cmb)/1.e26,label=r'$dE_{\rm th,oc}/dT_{\rm cmb}$',lw=2)
    ax3.set_xlabel(r'$T_{\rm cmb}$ (K)')
    ax3.set_ylabel(r'$dE/dT_{\rm cmb}$ ($10^{26}$J/K)')
    plt.legend(loc='upper left')
    plt.ylim(0.,2.5)
    ax3.text(1310,2.25,r'$R_{icb}=1325$ km',fontsize=28)
    ax3.text(1530,2.25,r'$R_{icb}=650$ km',fontsize=28)
    plt.savefig("materials/core_energetics.png")

    f4 = plt.figure()
    ax4 = plt.subplot(111)
    ax4.plot(t_cmb,dm_dT_cmb(t_cmb))
    ax4.set_xlabel('T_cmb (K)')
    ax4.set_ylabel('dM_ic/dT_cmb (kg/K)')

    plt.show()


    # Test 3: Wishlist quantities, fit quantaties as a function of r_icb and 
    # then return an interpolated quantity for

    model1.printData(['m_frac','r_frac','r_icb','r_cmb','L_m','Cp_ic',\
                'Cp_oc','w_bulk','w_l','w_s','P_cen','T_icb','T_cmb','P_icb','P_cmb',\
                'rho_cen','rho_liq_0','K_liq_0','alpha_t','alpha_c'] )

    # Chosen inner core radius
#     R = 650. * 1000 # 650 km
    R = 1325. * 1000 # 1325 km

    quants = ['m_frac','r_icb','r_cmb','L_m','Cp_oc','w_bulk','w_l','w_s','P_cen',\
                'T_cmb','P_cmb','rho_cen','rho_liq_0','K_liq_0','alpha_t','alpha_c']

    funcs = []
    vals = []
    for q in quants:
#         print q
        func = model1.func_of_ricb(q)
        val = float(func(R))
        funcs.append(func)
        vals.append(val)

    df = pd.DataFrame([vals])
    df.columns = quants
#     print df

    # Test 4: Wishlist quadratic fit to liquidus

    p_arr = model1.data.P_icb
    t_arr = model1.data.T_icb
    
    quadratic_const = np.polyfit(p_arr,t_arr,2)

#     ptest = 30.e9
#     print np.polyval(quadratic_const,ptest)
    df['Tm0'] = [ quadratic_const[-1] ]
    df['Tm1'] = [ quadratic_const[1] / quadratic_const[-1] ]
    df['Tm2'] = [ quadratic_const[0] / quadratic_const[-1] ]

    print df

# #     # determinine profiles at that snapshot for given core radius
# #     # print with the calues on the boundaries doubled (for seismology code)
# # 
#     fname='tables/elastic_06_650'
#     merc.generate_profiles(df.m_frac[0])
# #     merc.generate_profiles(0.)
# #     merc.show_profiles(fname='materials/profiles.png')
#     bounds = np.hstack((0.,merc.boundaries))
#     rlayer = bounds[1:] - bounds[:-1]
# 
#     N = 350
#     nsteps = np.floor(N * rlayer / sum(rlayer)).astype(int)
#     nreplace = [nsteps[0],nsteps[1]]
# 
# #     rstep = np.hstack([ np.linspace(x,y,z) for x,y,z in zip(bounds[:-1],bounds[1:],nsteps) ])
# 
#     partlist =[]
#     for i,layer in enumerate(merc.get_layers()):
#         if nsteps[i] == 0.:
#             print 'skip'
#             bounds[i+1] = 0.
#             continue
#         r0 = merc.radius[layer]
#         rho0 = merc.density[layer]
#         vp0 = merc.vp[layer]
#         vs0 = merc.vs[layer]
#         vphi0 = merc.vphi[layer]
#         K0 = merc.K[layer]
#         G0 = merc.G[layer]
# 
#         rhofunc = UnivariateSpline(r0,rho0,k=1)
#         vpfunc = UnivariateSpline(r0,vp0,k=1)
#         vsfunc = UnivariateSpline(r0,vs0,k=1)
#         vphifunc = UnivariateSpline(r0,vphi0,k=1)
#         Kfunc = UnivariateSpline(r0,K0,k=1)
#         Gfunc = UnivariateSpline(r0,G0,k=1)
# 
#         rstep = np.linspace(bounds[i],bounds[i+1],nsteps[i])
#         profile = pd.DataFrame()
#         profile['r'] = rstep
#         profile['rho'] = rhofunc(rstep)
#         profile['vp'] = vpfunc(rstep)
#         profile['vs'] = vsfunc(rstep)
#         profile['vphi'] = vsfunc(rstep)
#         profile['K'] = Kfunc(rstep)
#         profile['G'] = Gfunc(rstep)
# 
#         partlist.append(profile)
# 
#     profiles = pd.concat(partlist)
#     profiles.index = np.arange(len(profiles))
#     profiles[np.isnan(profiles)] = 0.
#     profiles[profiles < 0.] = 0.
# 
#     print profiles
#     print fname+'.csv'
#     profiles.to_csv(fname+'.csv')
# 
# #     vlist = merc.velocities_at_boundaries()
# #     for v,end in zip(vlist,['_icb.csv','_cmb.csv']):
# #         df1 = pd.DataFrame(v)
# #         df1.columns = profiles.columns
# #         print df1
# #         print fname+end
# #         df1.to_csv(fname + end)
#  
# plt.show()
