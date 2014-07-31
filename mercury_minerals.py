'''
mercury_minerals.py

Contains model material properties for a mercurian mantle and core.
'''

import burnman

# olar masses
mFe = 55.845
mSi = 28.0855
mS = 32.066

def value(x):
    return x

# Core material properties - These are unrealistic
# solids
class iron (burnman.Mineral):
    def __init__(self):
        '''
        Hauck2013
        mie-gruneisen from Boehler
        '''
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 6.830e-6, # 1 / (8.170 g / cm^3) * (.0558 kg / mol)
            'K_0': 165.0e9,
            'Kprime_0': 5.5,
            'G_0': 130.9e9, # where did shear stuff come from
            'Gprime_0': 1.92,
            'molar_mass': .0558,
            'n': 5.8,
            'Debye_0': 300.,
            'grueneisen_0': 1.66, # are these correct?
            'q_0': 0.6,
            'eta_s_0': 2.3 }

class iron_silicide17 (burnman.Mineral): 
    '''
    Hauck2013
    mie-gruneisen from Boehler (for Fe)
    '''
    def __init__(self):
        wSi = .17; wFe = 1.-wSi; mFe = 55.845; mSi = 28.0855
        xSi = (wSi/mSi) / ( wSi/mSi + wFe/mFe )
        xFe = 1. - xSi
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 6.828e-06, #  1 / (7.147 g / cm^3) * (.04880 kg / mol)
            'K_0': 199.0e9, 
            'Kprime_0': 5.7, 
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': .04880, #changed
            'n': 5.,
            'Debye_0': 300., # unknown?
            'grueneisen_0': 1.66, 
            'q_0': 0.6, # unknown ?
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
            'V_0': 6.92e-6, #1 / (8.069 g / cm^3) * (.0558 kg / mol)
            'K_0': 124.0e9,
            'Kprime_0': 5.5,
            'G_0': 0.e9,
            'Gprime_0': 0.,
            'molar_mass': .0558,
            'n': 1,
            'Debye_0': 10., # asymptote to 3R
            'grueneisen_0': 1.5, # alfe liquid iron
            'q_0': 0., # unkonwn
            'eta_s_0': 0.,
            } 
                                    
        
class iron_sulfide10_liquid (burnman.Mineral): 
    '''
    Fe - 10 wt % S endmember
    Balog 2003 
        A fit to the measured data only was made using
        density values obtained between 1.5 and 17.5 GPa at all
        three temperature levels considered, 1773 K, 1923 K, and
        2123 K. The third-order Birch-Murnaghan EOS revealed
        K0T = 64.3 GPa, K0'0T = 4.7, and 5.5 g/cm3 for the density of
        the Fe-10 wt % S at 1 atm. A
    '''
    def __init__(self):
        wS = .1; wFe = 1.-wS; #mFe = 55.845; mS = 32.066
        xS = (wS/mS) / ( wS/mS + wFe/mFe )
        xFe = 1. - xS
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 9.754e-06, #  1 / (5.33 g / cm^3) * (.05199 kg / mol)
            'K_0': 64.3e9, 
            'Kprime_0': 4.8, 
            'G_0': 0.,
            'Gprime_0': 0.,
            'molar_mass': .05199, 
            'n': 1,
            'Debye_0': 10., # assymptote to 3R
            'grueneisen_0': 1.5, #  Alfe liquid iron ?
            'q_0': 0., # unknown ?
            'eta_s_0': 0.,
            'mole_fraction' : xS,
            'weight_percent' : wS,
            'T0' : 1173}

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
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 8.394e-6, #  1 / (5.0 g / cm^3) * (.04197 kg / mol)
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
            'T_0' : 1450. # T_0 from dumay and cramb
            }
            
class iron_sulfide_liquid (burnman.Mineral): #placeholder
    def __init__(self):
        # Hauck 2013 / sanloup 2003 / Kairu and Toguri 1979
        xS = .5; xFe = 1. - xS
        wS = xS * mS / ( xS * mS + xFe * mFe)
        # wS = 0.36
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 1.127e-5, #  1 / (3.9 g / cm^3) * (.04396 kg / mol)
            'K_0': 0.1e9, # This takes a non-linear fit in Hauck 
                            # it will go unstable for large % sulfur
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
            'weight_percent' : wS,
            'T_0' : 1200.} # took T_0 directly from Kaiura and Toguri


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

class ironSilicideAlloy(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_Si):
        base_materials = [iron_liquid(), iron_silicide_liquid()]
        x0 = base_materials[1].params['mole_fraction']
        assert( mole_frac_Si <= x0 )
        molar_fraction = [1. - mole_frac_Si / x0, 0.0 + mole_frac_Si / x0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)

# liquid "alloys"

class ironSulfideLiquid(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_S):
        base_materials = [iron_liquid(), iron_sulfide10_liquid()]
        x0 = base_materials[1].params['mole_fraction']
        assert( mole_frac_S <= x0 )
        molar_fraction = [1. - mole_frac_S / x0, 0.0 + mole_frac_S / x0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)

class ironSilicideLiquid(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_Si):
        base_materials = [iron_liquid(), iron_silicide_liquid()]
        x0 = base_materials[1].params['mole_fraction']
        assert( mole_frac_Si <= x0 )
        molar_fraction = [1. - mole_frac_Si / x0, 0.0 + mole_frac_Si / x0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)

class ironSulfideSilicideLiquid(burnman.HelperSolidSolution):
    def __init__(self, mole_frac_S, mole_frac_Si):
        base_materials = [iron_liquid(),iron_sulfide10_liquid(),iron_silicide_liquid()]
        xS0 = base_materials[1].params['mole_fraction']
        xSi0 = base_materials[2].params['mole_fraction']
        assert( mole_frac_S <= xS0 )
        assert( mole_frac_Si <= xSi0 )
        molar_fraction = [1. - mole_frac_Si / xSi0 - mole_frac_S / xS0, 0.0 +
                mole_frac_S / xS0, 0.0 + mole_frac_Si / xSi0] # keep the 0.0 +, otherwise it is an array sometimes
        burnman.HelperSolidSolution.__init__(self, base_materials, molar_fraction)
