'''
mercury_minerals.py

Contains model material properties for a mercurian mantle and core.

Minerals are define using the
'''

import burnman
import numpy as np

import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline

from core_partition import x_to_w, w_to_x

# molar masses of elements
from mercury_reference import mFe,mSi,mS


# Core Material Properties
class gamma_iron(burnman.Mineral):
    def __init__(self):
        '''
        Parameters as presented in Dumbarry 2014 (Icarus)
        
        Parameters for gamma - Fe are from Tsujino et al. 2013
        Have no source for G_0 of Gprime_0 or eta_s_0.
        '''
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 1273.,
            'V_0': 7.381e-06, 
            'K_0': 111.5e9,
            'Kprime_0': 5.2,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': mFe / 1000.,
            'n': 1,
            'Debye_0': 340., 
            'grueneisen_0': 2.28, # are these correct?
            'q_0': 0.21,
            'eta_s_0': 0. }

class iron_silicide17(burnman.Mineral):
    def __init__(self):
        '''
        Lin 2003  hcp - Fe85Si15 up to 54 GPa at 300 K

        Note: I dont undestand why the pure iron reference (komabayashi)
        with a high K_0 was a problem and yet FeSi is 
        '''
        w = [0.,.17,1.-.17]; m = np.array([mS,mSi,mFe])
        x = w_to_x(w,m)
        molar_mass = np.sum( x * m ) / 1000.
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 300.,
            'V_0': 6.687e-06, 
            'K_0': 199.e9,
            'Kprime_0': 5.3,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': molar_mass,
            'n': 1,
            'Debye_0': 340., # using same as solid iron
            'grueneisen_0': 2.28, # using same as solid iron
            'q_0': 0.21, # using same as solid iron
            'eta_s_0': 0.,
            'mole_fraction' : x[1],
            'weight_percent' : w[1]} 


class iron_sulfide (burnman.Mineral): 
    def __init__(self):
        xS = .5
        xFe = 1. - xS
        wS = xS * mS / ( xS * mS + xFe * mFe)
        # Hauck 2013
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 9.991e-6, #  1 / (4.4 g / cm^3) * (.04396 kg / mol)
            'K_0': 54.3e9, 
            'Kprime_0': 4.,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': .04396, 
            'n': 2,
            'Debye_0': 300., # asymptote to 3R
            'grueneisen_0': 1.5,
            'q_0': 1.5,
            'eta_s_0': 0.,
            'mole_fraction': 0.,
            'weight_percent': 0.,
            'T_0' : 1000.} 

# Liquid Alloys

class liquid_iron(burnman.Mineral):
    def __init__(self):
        '''
        Parameters as presented in Dumbarry 2014 (Icarus)
        
        Parameters for liquid Fe are taken from Anderson and Ahrens (1994);

        Gruneisen parameter from low alpha and C_v = 3R
        85.4 GPa * 9.2e-5 K^-1 * 7.957 cm^3/mol / (3 R) = 2.506

        Gruneisen from low alpha and C_p from Desai1986
        85.4 GPa * 9.2e-5 K^-1 * 7.957 cm^3/mol / (36.216 J / mol / K) = 1.7262138

        Gruneisen from high alpha and C_p from Desai1986
        85.4 GPa * 13.2e-5 K^-1 * 7.957 cm^3/mol / (25.190 J / mol / K) = 3.5608444
        '''
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 1811.,
            'V_0': 7.957e-06, 
            'K_0': 85.3e9,
            'Kprime_0': 5.9, #5.9,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': mFe / 1000.,
            'n': 1,
            'Debye_0': 10., # C_v -> 3R
            'grueneisen_0': 1.7, # Calculated from alpha using C_v = 3R
            'q_0': 1.4,
            'eta_s_0': 0. }

class liquid_iron_sulfide10(burnman.Mineral):
    def __init__(self):
        '''
        Parameters as presented in Dumbarry 2014 (Icarus)
        
        parameters for l Fe10%S are from Balog et al. (2003)
        alpha from Kaiura and Toguri (1979) by applying Eq. (5).

        Gruneisen parameter from alpha:
        63. GPa * 10.e-5 K^-1 * 9.453 cm^3/mol / (3 R)

        Can this be extrapolated to allow higher S fractions
        '''
        w = [.1,0.,.9]; m = np.array([mS,mSi,mFe])
        x = w_to_x(w,m)
        molar_mass = np.sum( x * m ) / 1000.
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 1923.,
            'V_0': 9.453e-06, 
            'K_0': 63.0e9,
            'Kprime_0': 4.8,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': molar_mass,
            'n': 1,
            'Debye_0': 10., # C_v -> 3R
            'grueneisen_0': 1.7, # Calculated from alpha using C_v = 3R
            'q_0': 1.4,
            'eta_s_0': 0. ,
            'mole_fraction' : x[0],
            'weight_percent' : w[0]} 


class liquid_iron_sulfide20(burnman.Mineral):
    '''
    Extrapolation of model data from liquid_iron and liquid_iron_sulfide10
    to allow for simulations with higher S content. (from 10 to 20 wt. %)
    '''
    def __init__(self):
        w = [.2,0.,.8]; m = np.array([mS,mSi,mFe])
        x = w_to_x(w,m)
        molar_mass = np.sum( x * m ) / 1000.

        lFe = liquid_iron()
        lFeS10 = liquid_iron_sulfide10()

        p = 0.
        t = lFeS10.params['T_0']

        lFe.set_method('slb3');lFeS10.set_method('slb3')
        burnman.velocities_from_rock(lFe,np.array([p]),np.array([t]))
        burnman.velocities_from_rock(lFeS10,np.array([p]),np.array([t]))

        # mol fraction of each
        f =  w_to_x([.1,0.,.9],m)[0] / x[0]

        Kp0 = lFe.params['Kprime_0']; Kp10 = lFeS10.params['Kprime_0']

        V20  =  (  lFeS10.V - (1.-f)*lFe.V ) / f
        K20  = ( lFeS10.K_T - (1.-f)*lFe.K_T ) / f
#         Kp20  =  (  Kp0 - (1.-f)*Kp10 ) / f
        gamma20 = lFeS10.grueneisen_parameter()
        self.params = {
            'equation_of_state':'slb3',
            'T_0': t,
            'V_0': V20, 
            'K_0': K20,
            'Kprime_0': 3.7,# manually chose to reproduce 10 wt % result
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': molar_mass,
            'n': 1, 
            'Debye_0': 10., # C_v -> 3R
            'grueneisen_0': gamma20, 
            'q_0': 1.4,
            'eta_s_0': 0. ,
            'mole_fraction' : x[0],
            'weight_percent' : w[0]} 

class liquid_iron_silicide17(burnman.Mineral):
    def __init__(self):
        '''
        Yu and Secco Fe-17 Wt% (to 12 GPa)
        '''
        w = [0.,.17,1.-.17]; m = np.array([mS,mSi,mFe])
        x = w_to_x(w,m)
        molar_mass = np.sum( x * m ) / 1000.
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 1773.,
            'V_0': 1 / 5880. * molar_mass, 
            'K_0': 68.e9,
            'Kprime_0': 4.0,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': molar_mass,
            'n': 1,
            'Debye_0': 10., # C_v -> 3R
            'grueneisen_0': 2.5, # use same as liquid_iron
            'q_0': 1.4,
            'eta_s_0': 0. ,
            'mole_fraction' : x[1],
            'weight_percent' : w[1]} 



# Olivine and Orthopyroxene parameters from  stixrude & lithgow-bertelloni 2011
class forsterite (burnman.Mineral): 
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0':  4.360e-05,
            'K_0': 128.0e9,
            'Kprime_0': 4.2,
            'G_0': 82.9e9,
            'Gprime_0': 1.5,
            'molar_mass': .1406931, 
            'n': 7, 
            'Debye_0': 809.,
            'grueneisen_0': 0.99,
            'q_0': 2.0,
            'eta_s_0': 2.3 }

class fayalite (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 4.629e-5,
            'K_0': 135.0e9,
            'Kprime_0': 4.2,
            'G_0': 51.0e9,
            'Gprime_0': 1.5,
            'molar_mass': .2037731, 
            'n': 7,
            'Debye_0': 619.,
            'grueneisen_0': 1.06,
            'q_0': 3.6,
            'eta_s_0': 1.0 }
        
class enstatite (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0':  6.268e-05,
            'K_0': 107.0e9,
            'Kprime_0': 7.0,
            'G_0': 77.0e9,
            'Gprime_0': 1.5,
            'molar_mass': .2007774,
            'n': 10, 
            'Debye_0': 809.,
            'grueneisen_0': 0.78,
            'q_0': 3.4,
            'eta_s_0': 2.5 }

class ferrosillite (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0':  6.594e-05,
            'K_0': 101.0e9,
            'Kprime_0': 7.0,
            'G_0': 52.0e9,
            'Gprime_0': 1.5,
            'molar_mass': .2638574,
            'n': 10, 
            'Debye_0': 809.,
            'grueneisen_0': 0.72,
            'q_0': 3.4,
            'eta_s_0': 1.1 }

# Solid solutions
class olivine(burnman.HelperSolidSolution):
    def __init__(self, fe_num):
        base_materials = [forsterite(), fayalite()]
        molar_fraction = [1. - fe_num, 0.0 + fe_num] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)

class orthopyroxene(burnman.HelperSolidSolution):
    def __init__(self, fe_num):
        base_materials = [enstatite(), ferrosillite()]
        molar_fraction = [1. - fe_num, 0.0 + fe_num] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)


class  ironSilicideAlloy(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_Si):
        base_materials = [gamma_iron(), iron_silicide17()]
        x0 = base_materials[1].params['mole_fraction']
        assert( mole_frac_Si <= x0 )
        molar_fraction = [1. - mole_frac_Si / x0, 0.0 + mole_frac_Si / x0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)

# liquid "alloys"
class  ironSulfideSilicideLiquid(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_S,mole_frac_Si):
        # Fe, FeS and FeSi endmembers
        base_materials = \
            [liquid_iron(),liquid_iron_sulfide20(),liquid_iron_silicide17()]

        # check that composition isn't outside of the range of the model
        xS0 = base_materials[1].params['mole_fraction']
        xSi0 = base_materials[2].params['mole_fraction']
        assert( mole_frac_S <= xS0 )
        assert( mole_frac_Si <= xSi0 )

        molar_fraction = np.array([1. - mole_frac_Si / xSi0 - mole_frac_S / xS0, 
            0.0 + mole_frac_S / xS0, 0.0 + mole_frac_Si / xSi0] )
        # check
        assert molar_fraction[0] >= 0., "composition outside valid range"\
                +" for ternary mixing model"

        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)


def williams_adiabat(phase,t0,p):
#     t0 = 1500.
#     phase = liquid_iron()
    phase.set_method('slb3')
    phase.set_state(p[0],t0)
    rho0 = 1. / phase.V
    K0 = phase.K_T
    def williams_func(T,P):
        alpha0 = 13.2e-5
        Cp = 46.632 # J / mol / K
        phase.set_state(P,T)
        rho = 1. / phase.V
        K = phase.K_T
        return alpha0 * K0 / K * T / rho / Cp
#         return alpha0 * K0 / K * ( rho / rho0)**0.5*T/rho/Cp
    ad = np.ravel(integrate.odeint(williams_func,t0,p))
    return ad


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    lFe = liquid_iron()
    lFeS10 = liquid_iron_sulfide10()
    lFeS20 = liquid_iron_sulfide20()

    iron = gamma_iron()
    alloy = ironSilicideAlloy(0.)

    w = [.1,0.,.9]
    x = w_to_x(w)
    lFeS_ext = ironSulfideSilicideLiquid(x[0],0.)

#     # Testing densities along isotherm
#     p = np.linspace(0.,50.,101.) * 1.e9 # Pa
#     t0 = 1773.
#     fig1 = plt.figure()
#     ax1 = plt.subplot(111)
#     for phase in [ lFe,lFeS10,lFeS_ext,lFeS20]:
#         t = np.ones_like(p) * t0
#         phase.set_method('slb3')
#         rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(phase, p, t)
#         ax1.plot(p,rho)
# 
#     # Testing densities along adiabat
#     t0 = 1000.
#     fig2 = plt.figure()
#     ax2 = plt.subplot(111)
# 
#     fig3 = plt.figure()
#     ax3 = plt.subplot(111)
#     for phase in [ lFe,lFeS10,lFeS_ext,lFeS20, iron,alloy]:
#         phase.set_method('slb3')
#         t =burnman.geotherm.adiabatic(p,np.array([t0]),phase)
#         rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(phase, p, t)
#         ax2.plot(p,rho)
# #         ax3.plot(p,t)


    # creating a compoarison of dT/dP a la Williams 2009
#     from liquidus_model import Solver_no14 as FeSLiquidusModel
    from liquidus_model import Solver as FeSLiquidusModel
    from mercury_reference import Tm_anzellini
    p = np.linspace(0.,40.,101.) * 1.e9
    liquidus_at_P = lambda p: FeSLiquidusModel().T_SP(0.,p)
    liq = np.array([ liquidus_at_P(x) for x in p ])
    liq_anzellini = Tm_anzellini(p)
    dT_dP_liq = np.gradient(liq)/np.gradient(p) * 1.e9
    dT_dP_anzellini = np.gradient(liq_anzellini)/np.gradient(p) * 1.e9

    lFe_high = liquid_iron()
    lFe_high.params['grueneisen_0'] = 3.6
    lFe_low = liquid_iron()
    lFe_low.params['grueneisen_0'] = 1.7
    for ph in [lFe,lFe_high,lFe_low]: ph.set_method('slb3')

    # plot T of melting versus adiabats
    fig4 = plt.figure()
    ax4 = plt.subplot(111)
    thigh =burnman.geotherm.adiabatic(p,np.array([1700]),lFe_high)
    tlow  = burnman.geotherm.adiabatic(p,np.array([1900]),lFe_low)
    t  = burnman.geotherm.adiabatic(p,np.array([1800]),lFe)
    ax4.plot(p,liq,'k--') 
    ax4.plot(p,liq_anzellini,'k',lw=2) 
    ax4.plot(p,thigh,'g')
    ax4.plot(p,tlow,'r')
    ax4.plot(p,t,'b')

    # plot dT/dP of clapeyron slope versus adiabats
    fig5 = plt.figure()
    ax5 = plt.subplot(111)
    ax5.plot(p,dT_dP_liq,'k--') 
    ax5.plot(p,dT_dP_anzellini,'k',lw=2) 
    for phase,t0 in zip([lFe,lFe_high,lFe_low],[1800.,1700.,1900.]):
        phase.set_method('slb3')
        t =burnman.geotherm.adiabatic(p,np.array([t0]),phase)
        dT_dP_ad = np.gradient(t) / np.gradient(p) * 1.e9
        ax5.plot(p,dT_dP_ad)
    phase = lFe_high; lFe_high.params['Kprime_0'] = 7.
    t =burnman.geotherm.adiabatic(p,np.array([1700.]),phase)
    dT_dP_ad = np.gradient(t) / np.gradient(p) * 1.e9
    ax5.plot(p,dT_dP_ad,'g--')
    phase = lFe_high; lFe_high.params['Kprime_0'] = 4.6
    t =burnman.geotherm.adiabatic(p,np.array([1700.]),phase)
    dT_dP_ad = np.gradient(t) / np.gradient(p) * 1.e9
    ax5.plot(p,dT_dP_ad,'g-.')

    
    # plot Williams form of the adiabat
    lFe_high = liquid_iron()
    lFe_high.params['grueneisen_0'] = 3.6
    lFe_high.params['Kprime_0'] = 7.
    t = williams_adiabat(lFe_high,1700.,p)
    dT_dP_ad = np.gradient(t) / np.gradient(p) * 1.e9
    ax4.plot(p,t,'c')
    ax5.plot(p,dT_dP_ad,'c')


    plt.show()
