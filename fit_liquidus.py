'''
This code fits experimental data for the melting of a binary system.
The goal is to fit the liquidus to a polynomial model (after Stevenson 1983)
to describe the thermal evolotion of a parameterized convecting planet.

It contains the class 

    Liquidus()

Which reads in tabulated data on the Fe melting temperature and FeS
eutectic.

The relevent functional models are: 
---------------------------------------------------------------------

    Liquidus.Tliq_with_piecewise_eutectic(p,x)

    Liquidus.Tliq_simple(p,x)
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# fit to a specified polynomial form
from scipy.optimize import curve_fit

# fit to a univariate spine
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

class Liquidus(object):

    def __init__(self):
        eut = pd.read_csv('fes_liquidus_data/Chudinovskikh_fe_s_eutectic.csv',skiprows=5)
        eut.columns = [u'P',u'wt_pct',u'T']

        fe_melt = pd.read_csv('fes_liquidus_data/shen_fe_melting.csv',skiprows=5)        
        fe_melt.columns = [u'P',u'T']

        # convert GPa to Pa
        eut['P'] = eut['P'] * 1.e9
        fe_melt['P'] = fe_melt['P'] * 1.e9

        self.eut = eut
        self.fe_melt = fe_melt

        # convert wt. % to mole fraction
        mFe = 55.845
        mSi = 28.0855
        mS = 32.066

        wS = eut['wt_pct'] / 100.
        wFe = 1. - wS
        xS = (wS/mS) / ( wS/mS + wFe/mFe )

        self.eut['mol_frac'] = xS


        t_init = [ 1880. , 1.36e-12, -6.2e-24 ] 

        # functional fits
        self.Tm, params = self.func_from_fit(self.t_poly3,self.fe_melt,'P','T',t_init)
#         self.Teut, params = func_from_fit(t_poly3,eut,'P','T',t_init) 
#         Teut_sp = UnivariateSpline(eut['P'],eut['T'])

#         Xeut = UnivariateSpline(eut['P'],eut['mol_frac'])
#         Xeut_lin,params = func_from_fit(x_lin,eut,'P','mol_frac',None)

        # Try a piecewise fit for the eutectic
        self.eut1 = eut[:6]; self.eut2 = eut[6:]; self.pbreak = 10.e9
        self.Teut1 = interp1d(self.eut1['P'],self.eut1['T'],kind='linear')
        self.Teut2 = interp1d(self.eut2['P'],self.eut2['T'],kind='linear')

        self.Xeut1 = interp1d(self.eut1['P'],self.eut1['mol_frac'],kind='linear')
        self.Xeut2 = interp1d(self.eut2['P'],self.eut2['mol_frac'],kind='linear')

    # fitting wrapper
    def func_from_fit(self,func,data,xname,yname,p0=None):
        if p0 is None:
            fit = curve_fit(func,data[xname],data[yname])
        else:
            fit = curve_fit(func,data[xname],data[yname],p0)
        params = fit[0]
        f = lambda x: func(x,*params)
        return f, params

#     def piecewise_fit(self,func,data,xname,yname,xstep,p0=None):
#         
#         last = 0.
#         for x in xstep + [ dat[xname][-1] ]:
#             irange = (dat[xname] > last) & (dat[xname] <= x)
#             x_range = dat[irange]
# 
#             f, param = func_from_fit(func,dat,xname,yname,p0)

    # quadratic fit ( after Stevenson 1983)
    def t_poly3(self,p, t0, t1, t2):
        return t0 * ( 1. + t1 * p + t2 * p**2. )

    def x_lin(self,y,x0,x1):
        return x0 + x1*y


    def Teut_pw(self,p):
        p1 = p[ p <= self.pbreak]
        p2 = p[ p > self.pbreak]
        return np.hstack([self.Teut1(p1),self.Teut2(p2)])

    def Xeut_pw(self,p):
        p1 = p[ p <= self.pbreak]
        p2 = p[ p > self.pbreak]
        return np.hstack([self.Xeut1(p1),self.Xeut2(p2)])

    # build liquidus functions
    def Tliq_with_piecewise_eutectic(self,p,x):
        xfrac = x / self.Xeut_pw(p)
        return xfrac * self.Teut_pw(p) + (1. - xfrac) * self.Tm(p)
        
    def Tliq_simple(self,p,x):
        return (1. - 2.*x) * self.Tm(p)

if __name__ == "__main__":
    # plot and compare
    prange = np.linspace(0.e9,40.e9,100)

    # plt.figure()
    # plt.plot(fe_melt['P'],fe_melt['T'])
    # plt.plot(prange,Tm(prange))
    # 
    # plt.figure()
    # plt.plot(eut['P'],eut['T'])
    # plt.plot(prange,Teut(prange))
    # # plt.plot(prange,Teut_sp(prange))
    # plt.plot(prange,Teut_pw(prange))
    # 
    # plt.figure()
    # plt.plot(eut['P'],eut['mol_frac'])
    # # plt.plot(prange,Xeut_lin(prange))
    # plt.plot(prange,Xeut(prange))
    # plt.plot(prange,Xeut_pw(prange))

    x = 0.05
    liq = Liquidus()
    plt.figure()
    plt.plot(prange,liq.Tliq_with_piecewise_eutectic(prange,x) )
    plt.plot(prange,liq.Tliq_simple(prange,x) )
    plt.plot(prange,liq.Tm(prange),'k')
    plt.plot(prange,liq.Teut_pw(prange),'k')

    plt.show()


