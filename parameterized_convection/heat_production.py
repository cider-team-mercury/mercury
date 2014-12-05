class radioactive_species(object):
    def __init__(self, heat_release, half_life):
        self.half_life
        self.heat_release
    def heat_generation_rate(self,time):
        return self.heat_release*np.exp(-self.half_life*time)
# ------------------------------------------------------------- #
# - Heat Production and half-lives from Turcotte and Schubert - #
# ------------------------------------------------------------- #

uranium238 = radioactive_species(heat_release= 9.46e-5, half_life=4.47e9)
uranium235 = radioactive_species(heat_release= 9.69e-4, half_life=7.04e8)
thorium232 = radioactive_species(heat_release= 2.64e-5, half_life=1.40e10)
potasium40 = radioactive_species(heat_release= 2.92e-5, half_life=1.25e9)
# ------------------------------------------------------- #
