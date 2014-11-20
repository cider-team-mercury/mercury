'''
mercury_minerals.py

Contains model material properties for a mercurian mantle and core.
'''

import burnman
import numpy as np

import scipy.integrate as integrate
from scipy.interpolate import UnivariateSpline


# molar masses of elements
mFe = 55.845
mSi = 28.0855
mS = 32.066

# converting between mol and weight percent
def w_to_x(w1,m1=[mS,mSi,mFe]):
    '''
    Convert from mass fraction to mol fraction
    '''
    w = np.array(w1)
    m = np.array(m1)
    assert len(w) == len(m)

    x = (w / m) / np.sum(w / m)
    return x

def x_to_w(x1,m1=[mS,mSi,mFe]):
    '''
    Convert from mol fraction to mass fraction
    '''
    x = np.array(x1)
    m = np.array(m1)
    assert len(x) == len(m)

    w = (x * m) / np.sum(x * m)
    return w

def value(x):
    return x

# Core material properties - These are unrealistic
# solids
class iron (burnman.Mineral):
    def __init__(self):
        # Hauck2013
        self.params = {
            'equation_of_state':'slb3',
            'V_0': mFe / 1000. / 8170., #6.830e-6, # 1 / (8.170 g / cm^3) * (.0558 kg / mol)
            'K_0': 165.0e9,
            'Kprime_0': 5.5,
            'G_0': 130.9e9, # where did shear stuff come from
            'Gprime_0': 1.92,
            'molar_mass': mFe / 1000.,
            'n': 1,
            'Debye_0': 474., #Poirier book
            'grueneisen_0': 1.5, # are these correct?
            'q_0': 1.5,
            'eta_s_0': 2.3 }

class iron_dumberry(burnman.Mineral):
    def __init__(self):
        # Hauck2013
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
            'Debye_0': 340., #Poirier book
            'grueneisen_0': 2.28, # are these correct?
            'q_0': 0.21,
            'eta_s_0': 0. }

class liquid_iron_dumberry(burnman.Mineral):
    def __init__(self):
        # Hauck2013
        self.params = {
            'equation_of_state':'slb3',
            'T_0': 1811.,
            'V_0': 7.957e-06, 
            'K_0': 85.3e9,
            'Kprime_0': 5.9,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': mFe / 1000.,
            'n': 1,
            'Debye_0': 10., # C_v -> 3R
            'grueneisen_0': 1.5, # From alfe?
            'q_0': 1.4,
            'eta_s_0': 0. }

class liquid_iron_sulfide10_dumberry(burnman.Mineral):
    def __init__(self):
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
            'grueneisen_0': 1.5, # From alfe?
            'q_0': 1.4,
            'eta_s_0': 0. ,
            'mole_fraction' : x[0],
            'weight_percent' : w[0]} 


class iron_silicide17 (burnman.Mineral): 
    # Hauck 2013
    def __init__(self):
        wSi = .17; wFe = 1.-wSi; mFe = 55.845; mSi = 28.0855
        xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe )
        xFe = 1. - xSi
        mol_mass = xSi*mSi + xFe*mFe /1000.
        self.params = {
            'equation_of_state':'slb3',
            'V_0': mol_mass / 7147., #6.828e-06, #  1 / (7.147 g / cm^3) * (.04880 kg / mol)
            'K_0': 199.0e9, 
            'Kprime_0': 5.7, 
            'G_0': 130.9e9, # where did shear stuff come from
            'Gprime_0': 1.92,
            'molar_mass': mol_mass, #changed
            'n': 1,
            'Debye_0': 474., # poirier book
            'grueneisen_0': 1.5, 
            'q_0': 1.5, # unknown ?
            'eta_s_0': 0.,
            'mole_fraction' : xSi,
            'weight_percent' : wSi}

class iron_sulfide (burnman.Mineral): 
    def __init__(self):
        xS = .5
        xFe = 1. - xS
        wS = xS * mS / ( xS * mS + xFe * mFe)
#         print xS, wS
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

# liquids

class iron_liquid (burnman.Mineral): 
    # Hauck 2013
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0': mFe / 1000. / 8060., # 6.92e-6, #1 / (8.069 g / cm^3) * (.0558 kg / mol)
            'K_0': 124.0e9,
            'Kprime_0': 5.5,
            'G_0': 0.e9,
            'Gprime_0': 0.,
            'molar_mass': .0558,
            'n': 1,
            'Debye_0': 10., # asymptote to 3R
            'grueneisen_0': 1.5, # alfe liquid iron
            'q_0': 0., # unkonwn
            'eta_s_0': 0.}
                                    
class iron_silicide_liquid (burnman.Mineral): 
    '''
    Fe-Si liquid endmember
    Parameters from:
        Hauck et. al 2013
        T_0 and V_0 from dumay and cramba 1995
    '''
    def __init__(self):
        #Hauck 2013 / dumay and cramb
        xSi = .5; xFe = 1. - xSi
        wSi = xSi * mSi / ( xSi * mSi + xFe * mFe)

        # convert volume to one consistent at 300 K.
        T0 = 1450.; alpha0 = 9.2e-5; VatRef = 8.394e-6
        therm_exp = lambda v,t : v * alpha0
        volume = np.ravel(integrate.odeint(therm_exp,VatRef,[T0,300.]) )

        self.params = {
            'equation_of_state':'slb3',
            'V_0': volume[-1], #  1 / (5.0 g / cm^3) * (.04197 kg / mol)
            'K_0': 84.0e9, # took value of -80 dK/d (124 - 40)
            'Kprime_0': 5.5,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': .04197,
            'n': 1,
            'Debye_0': 10., # asymptote to 3R
           'grueneisen_0': 1.5,
            'q_0': 0.,
            'eta_s_0': 0.,
            'mole_fraction' : xSi,
            'weight_percent' : wSi,
            }
            
class iron_sulfide_liquid (burnman.Mineral): #placeholder
    '''
    Fe-S liquid endmember
    Parameters from:
        Hauck 2013.
        K_0 not included, see ironSulfideSilicideLiquid().
        T0 from Kaiura and Toguri.
    '''
    def __init__(self):
        # Hauck 2013 / sanloup 2003 / Kairu and Toguri 1979
        xS = .5; xFe = 1. - xS
        wS = xS * mS / ( xS * mS + xFe * mFe)

        # convert Volume to 300. for V0
        T0 = 1200.; alpha0 = 1.1e-4; VatRef = 1.127e-5
        therm_exp = lambda v,t : v * alpha0
        volume = np.ravel(integrate.odeint(therm_exp,VatRef,[T0,300.]) )

        self.params = {
            'equation_of_state':'slb3',
            'V_0': volume[-1], #  1 / (3.9 g / cm^3) * (.04396 kg / mol)
            'K_0': np.nan, # see ironSulfideSilicideLiqui()
            'Kprime_0': 5.,
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': .04396, 
            'n': 1,
            'Debye_0': 10., # asymptote to 3R
            'grueneisen_0': 1.5,
            'q_0': 0.,
            'eta_s_0': 0.,
            'mole_fraction' : xS,
            'weight_percent' : wS} 


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
        base_materials = [iron_liquid(), iron_silicide_liquid()]
        x0 = base_materials[1].params['mole_fraction']
        assert( mole_frac_Si <= x0 )
        molar_fraction = [1. - mole_frac_Si / x0, 0.0 + mole_frac_Si / x0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)
        self.method = 'slb3'

# liquid "alloys"

class ironSulfideSilicideLiquid(burnman.Mineral):
    '''
    Simple model for a Fe-S-Si liquid alloy using parameters aggregated by Hauck
    et. al 2013.

    Uses parametrization of K from Sanloup 2002, assuming that K depends only on
    S content, while rho_0 is a linear mixture between Fe, FeS and FeSi endmembers.
    '''
    def __init__(self, mole_frac_S, mole_frac_Si):

        # Fe, FeS and FeSi endmembers
        base_materials = [iron_liquid(),iron_sulfide_liquid(),iron_silicide_liquid()]

        # check that composition isn't outside of the range of the model
        xS0 = base_materials[1].params['mole_fraction']
        xSi0 = base_materials[2].params['mole_fraction']
        assert( mole_frac_S <= xS0 )
        assert( mole_frac_Si <= xSi0 )

        # molar fraction of each component to include
        molar_fraction = np.array([1. - mole_frac_Si / xSi0 - mole_frac_S / xS0, 
            0.0 + mole_frac_S / xS0, 0.0 + mole_frac_Si / xSi0] )
        print molar_fraction
        wS = mole_frac_S * mS / ( mole_frac_S * mS + mole_frac_Si * mSi +
                (1.-mole_frac_S-mole_frac_Si)*mFe )

        # all other parameters take to be the same as pure liquid iron
        self.params = base_materials[0].params

        # linearly averaged parameters
        for param in ['V_0','molar_mass']:
            end_members = np.array([ mat.params[param] for mat in base_materials])
            print end_members
            self.params[param] = np.sum(molar_fraction * end_members)

        # use FeS K0 model from sanloup 2002
        KFeSpoly = lambda w: ( w**2 * 554. + w * -391. + 83.1  ) * 1.e9
        self.params['K_0'] = KFeSpoly(wS)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # solid mantle
    n_fe_ol = 0.5
    n_fe_opx = 0.5
    ol = olivine(n_fe_ol)
    opx = orthopyroxene(n_fe_opx)

    fol = 0.2; fopx = 1. - fol
    rock = burnman.Composite([fol,fopx],[ol,opx])

    # solid inner core / FeS layer
    wSi = 0.05; wFe = 1.-wSi
    xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe )
#     iron = iron()
    iron = iron_dumberry()
    solidFeS = iron_sulfide() # solid FeS
    solidFeSi = ironSilicideAlloy(xSi) # solid solution of Si in Fe

    # liquid outer core
    wS = 0.0; wSi = 0.0; wFe = 1.-wS - wSi
    xS = (wS/mS) / ( wS/mS + wFe/mFe + wSi/mSi)
    xSi = (wSi/mSi) / ( wS/mS + wFe/mFe + wSi/mSi)

#     liquidFe = iron_liquid() # pure liquid Fe
    liquidFe = liquid_iron_dumberry()
#     liquidFeS = ironSulfideLiquid(xS) # solution of liquid Fe and FeS
#     liquidFeSi = ironSilicideLiquid(xSi) # solution of FeSi and Fe
    liquidFeSSi = ironSulfideSilicideLiquid(xS,xSi) # ternary solution

    p = np.linspace(0.,50.,101.) * 1.e9 # Pa
    t0 = 1500.
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    for phase in [ iron,liquidFe]:
        t = np.ones_like(p) * t0
        phase.set_method('slb3')
        rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(phase, p, t)
        ax1.plot(p,rho)

    t0 = 1000.
    fig2 = plt.figure()
    ax2 = plt.subplot(111)

    fig3 = plt.figure()
    ax3 = plt.subplot(111)
    for phase in [ iron,liquidFe]:
        t =burnman.geotherm.adiabatic(p,np.array([t0]),phase)
        phase.set_method('slb3')
        rho, vp, vs, vphi, K, G = burnman.velocities_from_rock(phase, p, t)
        ax2.plot(p,rho)
        ax3.plot(p,t)

    plt.show()


    plt.show()



# Calculate gruneisen parameters 
# You have: 85.4 GPa * 9.2e-5 K^-1 * 7.957 cm^3/mol / (3 R)
# You want:
# Definition: 2.506335
# You have: 63. GPa * 10.e-5 K^-1 * 9.453 cm^3/mol / (3 R)
# You want:
# Definition: 2.3875599
