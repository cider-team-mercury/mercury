'''
This code fits experimental data for the melting of a binary system.
The goal is to fit the liquidus to a polynomial model (after Stevenson 1983)
to describe the thermal evolotion of a parameterized convecting planet.
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# fit to a specified polynomial form
from scipy.optimize import curve_fit

# fit to a univariate spine
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

eut = pd.read_csv('Chudinovskikh_fe_s_eutectic.csv',skiprows=5)
eut.columns = [u'P',u'wt_pct',u'T']

fe_melt = pd.read_csv('shen_fe_melting.csv',skiprows=5)        
fe_melt.columns = [u'P',u'T']

# convert GPa to Pa
eut['P'] = eut['P'] * 1.e9
fe_melt['P'] = fe_melt['P'] * 1.e9

# convert wt. % to mole fraction
mFe = 55.845
mSi = 28.0855
mS = 32.066

wS = eut['wt_pct'] / 100.
wFe = 1. - wS
xS = (wS/mS) / ( wS/mS + wFe/mFe )

eut['mol_frac'] = xS

# fitting wrapper
def func_from_fit(func,data,xname,yname,p0=None):
    if p0 is None:
        fit = curve_fit(func,data[xname],data[yname])
    else:
        fit = curve_fit(func,data[xname],data[yname],p0)
    params = fit[0]
    f = lambda x: func(x,*params)
    return f, params

def piecewise_fit(func,data,xname,yname,xstep,p0=None):
    
    last = 0.
    for x in xstep + [ dat[xname][-1] ]:
        irange = (dat[xname] > last) & (dat[xname] <= x)
        x_range = dat[irange]

        f, param = func_from_fit(func,dat,xname,yname,p0)



# quadratic fit ( after Stevenson 1983)
def t_poly3(p, t0, t1, t2):
    return t0 * ( 1. + t1 * p + t2 * p**2. )

def x_lin(y,x0,x1):
    return x0 + x1*y

t_init = [ 1880. , 1.36e-12, -6.2e-24 ] 

# functional fits
Tm, params = func_from_fit(t_poly3,fe_melt,'P','T',t_init)
Teut, params = func_from_fit(t_poly3,eut,'P','T',t_init) 
Teut_sp = UnivariateSpline(eut['P'],eut['T'])

Xeut = UnivariateSpline(eut['P'],eut['mol_frac'])
Xeut_lin,params = func_from_fit(x_lin,eut,'P','mol_frac',None)

# Try a piecewise fit for the eutectic
eut1 = eut[:6]; eut2 = eut[6:]; pbreak = 10.e9
Teut1 = interp1d(eut1['P'],eut1['T'],kind='linear')
Teut2 = interp1d(eut2['P'],eut2['T'],kind='linear')

Xeut1 = interp1d(eut1['P'],eut1['mol_frac'],kind='linear')
Xeut2 = interp1d(eut2['P'],eut2['mol_frac'],kind='linear')

def Teut_pw(p):
    p1 = p[ p <= pbreak]
    p2 = p[ p > pbreak]
    return np.hstack([Teut1(p1),Teut2(p2)])

def Xeut_pw(p):
    p1 = p[ p <= pbreak]
    p2 = p[ p > pbreak]
    return np.hstack([Xeut1(p1),Xeut2(p2)])

# build liquidus functions
def Tliq_with_piecewise_eutectic(p,x):
    xfrac = x / Xeut_pw(p)
    return xfrac * Teut_pw(p) + (1. - xfrac) * Tm(p)
    
def Tliq_simple(p,x):
    return (1. - 2.*x) * Tm(p)

# plot and compare
prange = np.linspace(eut['P'].min(),eut['P'].max(),100)

plt.figure()
plt.plot(fe_melt['P'],fe_melt['T'])
plt.plot(prange,Tm(prange))

plt.figure()
plt.plot(eut['P'],eut['T'])
plt.plot(prange,Teut(prange))
# plt.plot(prange,Teut_sp(prange))
plt.plot(prange,Teut_pw(prange))

plt.figure()
plt.plot(eut['P'],eut['mol_frac'])
# plt.plot(prange,Xeut_lin(prange))
plt.plot(prange,Xeut(prange))
plt.plot(prange,Xeut_pw(prange))

x = 0.05
plt.figure()
plt.plot(prange,Tliq_with_piecewise_eutectic(prange,x) )
plt.plot(prange,Tliq_simple(prange,x) )

plt.show()


