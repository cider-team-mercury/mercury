'''
This code fits experimental data for the melting of a binary system.
The goal is to fit the liquidus to a polynomial model (after Stevenson 1983)
to describe the thermal evolotion of a parameterized convecting planet.
'''

import numpy as np
import pandas as pd

# fit to a specified polynomial form
from scipy.optimize import curve_fit

# fit to a univariate spine
from scipy.interpolate import UnivariateSpline

eut = pd.read_csv('Chudinovskikh_fe_s_eutectic.csv',skiprows=5)
eut.columns = [u'P',u'wt_pct',u'T']

fe_melt = pd.read_csv('shen_fe_melting.csv',skiprows=5)        
fe_melt.columns = [u'P',u'T']

# convert GPa to Pa
eut['P'] = eut['P'] * 1.e9
fe_melt['P'] = fe_melt['P'] * 1.e9
