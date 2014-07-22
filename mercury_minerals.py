'''
mercury_minerals.py

Contains model material properties for a mercurian mantle and core.
'''

# Core material properties - These are unrealistic
class iron (burnman.Mineral):
    def __init__(self):
        self.params = {
            'equation_of_state':'slb3',
            'V_0': 6.6e-6,
            'K_0': 180.0e9,
            'Kprime_0': 4.9,
            'G_0': 130.9e9,
            'Gprime_0': 1.92,
            'molar_mass': .0558,
            'n': 1,
            'Debye_0': 300.,
            'grueneisen_0': 1.5,
            'q_0': 1.5,
            'eta_s_0': 2.3 }

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
