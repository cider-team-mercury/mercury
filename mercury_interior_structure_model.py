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
            print self.detect_snow()
            print self.adiabat_steeper()

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


if __name__ == "__main__":
    fig_size = [1000/72.27 ,800/72.27]
    params = {'backend': 'ps', 'axes.labelsize': 22, 'text.fontsize': 22,
            'legend.fontsize': 18,
              'xtick.labelsize': 16, 'ytick.labelsize': 16, 
              'xtick.major.size': 10,'ytick.major.size': 10,
              'xtick.minor.size': 6,'ytick.minor.size': 6,
              'xtick.major.width': 2,'ytick.major.width': 2,
              'xtick.minor.width': 2,'ytick.minor.width': 2,
              'axes.linewidth': 2, 'xaxis.labelpad' : 50,
              'text.usetex': False, 'figure.figsize': fig_size,
              'figure.subplot.bottom': 0.100,'figure.subplot.top': 0.980,'figure.subplot.left': 0.130,'figure.subplot.right': 0.950}
    plt.rcParams.update(params)
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
    model1.get_energetics()
    model1.printData()
    model1.saveData('tables/highres2_63_06_00.dat')

    # Load results from a saved model suite.
#     model1.loadData('tables/energetics_63_06_00.dat')
#     model1.printData()


#     ### Test 1: Look at profiles and determine whether snow predicted
#     merc.generate_profiles(0.5)
#     merc.show_profiles()

#     ### Test 2: Fit functions of Tcmb and plot energetic quantities

    r_func = model1.func_of_Tcmb('r_icb',s=1.e8 ) # m
#     r_func = model1.func_of_Tcmb('r_icb',k=2) # m

    dr_icb_dT_cmb = r_func.derivative() # m / K
    #wrap to get rid of unrealistic negative quantities
    def Eg_r_func(temp):
        func =  model1.func_of_Tcmb('Eg_r',s=0)
        t2 = temp[ temp > model1.data.T_cmb[1]]
        t1 = temp[ temp <= model1.data.T_cmb[1]]

        en2  = float(func(model1.data.T_cmb[1])) * ( model1.data.T_cmb[0] - t2) \
                        / (model1.data.T_cmb[0] - model1.data.T_cmb[1])
        en1 = func(t1)
        return np.hstack((en1,en2))

            
    def L_r_func(temp):
        func =  model1.func_of_Tcmb('L_r',s=0)
        t2 = temp[ temp > model1.data.T_cmb[1]]
        t1 = temp[ temp <= model1.data.T_cmb[1]]

        en2  = float(func(model1.data.T_cmb[1])) * ( model1.data.T_cmb[0] - t2) \
                        / (model1.data.T_cmb[0] - model1.data.T_cmb[1])
        en1 = func(t1)
        return np.hstack((en1,en2))


#     Eg_m_func = model1.func_of_Tcmb('Eg_m')
#     L_m_func = model1.func_of_Tcmb('L_m')

    dEth_ic, dEth_oc = model1.thermal_energy_change(s=1.e8)

    t_icb = np.linspace(model1.data.T_icb.min(),model1.data.T_icb.max(),100)
    t_cmb = np.linspace(model1.data.T_cmb.min(),model1.data.T_cmb.max(),100)


    f1 = plt.figure()
    ax1= plt.subplot(111)
#     ax1.plot(t_icb,model1.func_of_Ticb('r_icb')(t_icb))
#     ax1.plot(t_cmb,r_func(t_cmb))
    ax1.plot(t_cmb,r_func(t_cmb))
    ax1.plot(model1.data.T_cmb,model1.data.r_icb,'bo')
    ax1.set_xlabel('T_cmb (K)')
    ax1.set_ylabel('dR_icb (m)')


    # check units on these
    f2 = plt.figure()
    ax2 = plt.subplot(111)
    ax2.plot(t_cmb,Eg_r_func(t_cmb))
    ax2.plot(t_cmb,L_r_func(t_cmb))
    ax2.plot(model1.data.T_cmb,model1.data.Eg_r,'bo')
    ax2.plot(model1.data.T_cmb,model1.data.L_r,'go')
#     ax2.plot(t_cmb,dEth_ic(t_cmb))
#     ax2.plot(t_cmb,dEth_oc(t_cmb))
    ax2.set_xlabel('T_cmb (K)')
    ax2.set_ylabel('dE/dR_icb (J/m)')

    f3 = plt.figure()
    ax3 = plt.subplot(111)
    ax3.plot(t_cmb,-Eg_r_func(t_cmb)*dr_icb_dT_cmb(t_cmb),label='Eg')
    ax3.plot(t_cmb,-L_r_func(t_cmb)*dr_icb_dT_cmb(t_cmb),label='L')
    ax3.plot(t_cmb,dEth_ic(t_cmb),label='Cpt_ic')
    ax3.plot(t_cmb,dEth_oc(t_cmb),label='Cpt_oc')
    ax3.set_xlabel('T_cmb (K)')
    ax3.set_ylabel('dE/dT_cmb (J/K)')
    plt.legend()

    f4 = plt.figure()
    ax4 = plt.subplot(111)
    ax4.plot(t_cmb,dr_icb_dT_cmb(t_cmb))
    ax4.set_xlabel('T_cmb (K)')
    ax4.set_ylabel('dR_icb/dT_cmb (m/K)')

#     f3 = plt.figure()
#     ax3 = plt.subplot(111)
#     ax3.plot(t_cmb,Eg_m_func(t_cmb))
#     ax3.plot(t_cmb,L_m_func(t_cmb))

#     f4 = plt.figure()
#     ax4 = plt.subplot(111)
#     ax4.plot(t,model1.func_of_Ticb('CpT_avg_ic')(t))
#     ax4.plot(t,model1.func_of_Ticb('CpT_avg_oc')(t))
# 
#     f4 = plt.figure()
#     ax4 = plt.subplot(111)
#     ax4.plot(t,model1.func_of_Ticb('CpT_avg_ic').derivative()(t))
#     ax4.plot(t,model1.func_of_Ticb('CpT_avg_oc').derivative()(t))

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

#     # determinine profiles at that snapshot for given core radius
# 
#     fname='tables/elastic_09_0.csv'
#     merc.generate_profiles(df.m_frac[0])
#     merc.show_profiles(fname='materials/profiles.png')
#     merc.generate_profiles(0.)
#     r = merc.radius
#     rho = merc.density
#     vp = merc.vp
#     vs = merc.vs
#     K = merc.K
#     G = merc.G
#     vs[np.isnan(vs)] = 0. # Filter out nonsense
#     G[G < 0.] = 0.
# 
#     profiles = pd.DataFrame()
#     profiles['r'] = r
#     profiles['rho'] = rho
#     profiles['vp'] = vp
#     profiles['vs'] = vs
#     profiles['K'] = K
#     profiles['G'] = G
# 
#     print profiles
#     profiles.to_csv(fname)



