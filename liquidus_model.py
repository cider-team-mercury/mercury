'''
Model of the FeS eutectic liquidus courtousy of Nicholas Knezek with 
slight modifications.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import burnman
import burnman.minerals as minerals
import burnman.mineral_helpers as bmb
import burnman.composite as composite
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline

from core_partition import w_to_x,x_to_w

from mercury_reference import Tm_anzellini

from scipy.optimize import fsolve

def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

class Solver:
    def __init__(self):
        '''
        Class to calculate Eutectic liquidus for Fe/FeS mixtures
        '''
        # Low Sulfur Freeze P-T-S Values
#         S_l = [0.,3.,6.,9.,12.]
        P_l = [0.0,10.,14.,23.,40.]

        self.pressures = np.array([3.,10.,14.,21.,23.,30.,40.])


        S_3GPa_l = np.array([0.,     1.25254, 2.16308, 3.07288, 4.09596, 5.34501,
                6.82115, 8.18293, 9.31780, 10.2258, 11.3606, 12.6099, 13.8591,
                15.3362, 16.7002, 17.6096, 18.8599, 19.8834, 21.1348, 22.2730,
                23.5263, 24.8935, 25.6926, 26.3774])

        T_3GPa_l = np.array([1608.31, 1567.11, 1539.67, 1516.80, 1493.95, 1474.55,
                1451.72, 1435.77, 1422.09, 1410.68, 1397.01, 1376.46, 1355.91,
                1327.35, 1297.64, 1277.07, 1249.65, 1224.50, 1190.20, 1155.89,
                1110.13, 1059.79, 1020.88, 988.831])
        S_10GPa_l = np.array([0.      , 1.09756, 2.31707, 3.29268, 4.26829, 5.36585,
                6.82927, 8.17073, 9.26829, 9.87805, 10.9756, 12.1951, 13.2927,
                14.3902, 15.8537, 16.9512, 17.8049, 18.6585, 19.5122,
                20.,20.4878,25.])
        T_10GPa_l = np.array([2115.18, 2075, 2025.89, 1990.18, 1954.46, 1918.75,
                1874.11,1825, 1789.29, 1766.96, 1726.79, 1677.68, 1633.04,
                1583.93,1508.04, 1441.07, 1387.50, 1333.93, 1271.43, 1222.32,
                1191.07,700.])
        S_14GPa_l = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,18.27,])
        T_14GPa_l = np.array([2219.,2071.,1954.,1863.,1793.,1743.,1708.,1685.,1669.,1658.,1648.,1635.,1615.,1585.,1542.,1482.,1400.,1294.,1160.,1119.,])
        S_21GPa_l = np.array([0.,3.,6.,9.,12.,15.4])
        T_21GPa_l = np.array([2271.,2185.,2081.,1949.,1758.,1348.])
        S_23GPa_l = np.array([0.,3.,6.,9.,12.,16.])
        T_23GPa_l = np.array([2308.,2211.,2106.,1985.,1819.,1435.])
        S_40GPa_l = np.array([0.,3.,6.,9.,12.])
        T_40GPa_l = np.array([2498.,2372.,2215.,1997.,1565.])

        # modify the pure iron value to be consistent with Anzellini

        # Find T freeze for 30 GPa
        x = [23.,40.]
        S_30GPa_l = np.array([0.,3.,6.,9.,12.])
        T_30GPa_l = np.empty_like(S_40GPa_l)
        T23_for_loop = [2308.,2211.,2106.,1985.,1819.]
        for i,(T40,T23) in enumerate(zip(T_40GPa_l,T23_for_loop)):
                T_30GPa_l[i] = np.interp(30.,x,[T23, T40])

        # modify the pure iron melting temperature to be consistent with Anzellini
        for p,arr in zip([3.,10.,14.,21.,23.,30.,40.],\
                        [T_3GPa_l, T_10GPa_l,T_14GPa_l,T_21GPa_l,T_23GPa_l,\
                        T_30GPa_l,T_40GPa_l]):
            Tm_norm = Tm_anzellini(p*1.e9)
            Tm_0 = arr[0]
            Tl = arr[-1]
            arr[::-1] = (arr[::-1] - Tl) * (Tm_norm - Tl) / (Tm_0 - Tl) + Tl 


        # High Sulfur P-T-S Values
        S_3GPa_h = np.array([26.3774,27.1670, 28.5210, 29.8744, 31.3417, 32.8094, 
                33.8255,35.0685])

        T_3GPa_h = np.array([988.831,1009.51, 1041.69, 1078.46, 1110.64, 1140.54, 
                1161.23, 1179.65])
        S_10GPa_h = np.array([20.6098, 22.0732, 24.5122, 27.6829, 31.0976, 33.4146,
                36.8293])

        T_10GPa_h = np.array([1200., 1298.21, 1441.07, 1548.21, 1646.43, 1704.46, 1775.89])
        S_14GPa_mid = np.array([19.,20.,20.8])
        T_14GPa_mid = np.array([1144.,1160.,1173.])
        S_14GPa_h = np.array([21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36])
        T_14GPa_h = np.array([1236.,1394.,1471.,1528.,1576.,1618.,1656.,1690.,1722.,1752.,1781.,1808.,1834.,1858.,1882.,1905.])
        S_21GPa_h = np.array([16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,])
        T_21GPa_h = np.array([1370.,1405.,1428.,1443.,1454.,1461.,1467.,1472.,1477.,1480.,1532.,1586.,1637.,1685.,1731.,1775.,1818.,1859.,1898.,1937.,1974.,])
        S_23GPa_h = np.array([16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,])
        T_23GPa_h = np.array([1435.,1450.,1461.,1470.,1485.,1492.,1555.,1614.,1669.,1720.,1768.,1812.,1852.,1888.,1921.,1950.,1976.,1997.,2015.,2029.,2040.,])
        S_30GPa_h = np.array([14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,])
        T_30GPa_h = np.array([1500.,1575.,1646.,1707.,1759.,1804.,1842.,1876.,1906.,1934.,1958.,1982.,2003.,2023.,2042.,2060.,2077.,2093.,2109.,2124.,2138.,2151.,2164.,])
        S_40GPa_h = np.array([13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.,25.,26.,27.,28.,29.,30.,31.,32.,33.,34.,35.,36.,])
        T_40GPa_h = np.array([1647.,1730.,1787.,1833.,1873.,1908.,1941.,1971.,1999.,2026.,2051.,2075.,2098.,2120.,2142.,2162.,2182.,2202.,2220.,2239.,2256.,2274.,2291.,2308.,])

        S_21GPa_h1 = S_21GPa_h[0:10]
        S_21GPa_h2 = S_21GPa_h[10:-1]
        T_21GPa_h1 = T_21GPa_h[0:10]
        T_21GPa_h2 = T_21GPa_h[10:-1]
        S_23GPa_h1 = S_23GPa_h[0:5]
        S_23GPa_h2 = S_23GPa_h[5:-1]
        T_23GPa_h1 = T_23GPa_h[0:5]
        T_23GPa_h2 = T_23GPa_h[5:-1]

        self.p3lsp = UnivariateSpline(S_3GPa_l,T_3GPa_l,k=2)
        self.p10lsp = UnivariateSpline(S_10GPa_l,T_10GPa_l,k=2)
        self.p14lsp = UnivariateSpline(S_14GPa_l,T_14GPa_l,k=2)
        self.p21lsp = UnivariateSpline(S_21GPa_l,T_21GPa_l,k=2)
        self.p23lsp = UnivariateSpline(S_23GPa_l,T_23GPa_l,k=2)
        self.p30lsp = UnivariateSpline(S_30GPa_l,T_30GPa_l,k=2)
        self.p40lsp = UnivariateSpline(S_40GPa_l,T_40GPa_l,k=2)
        self.p14msp = UnivariateSpline(S_14GPa_mid,T_14GPa_mid,k=2)
        self.p3hsp = UnivariateSpline(S_3GPa_h,T_3GPa_h,k=2)
        self.p10hsp = UnivariateSpline(S_10GPa_h,T_10GPa_h,k=2)
        self.p14hsp = UnivariateSpline(S_14GPa_h,T_14GPa_h,k=2)
        self.p21hsp1 = UnivariateSpline(S_21GPa_h1,T_21GPa_h1,k=2)
        self.p21hsp2 = UnivariateSpline(S_21GPa_h2,T_21GPa_h2,k=2)
        self.p23hsp1 = UnivariateSpline(S_23GPa_h1,T_23GPa_h1,k=2)
        self.p23hsp2 = UnivariateSpline(S_23GPa_h2,T_23GPa_h2,k=2)
        self.p30hsp = UnivariateSpline(S_30GPa_h,T_30GPa_h,k=2)
        self.p40hsp = UnivariateSpline(S_40GPa_h,T_40GPa_h,k=2)

        self.find_eutectic()

    '''
    Define Functions to find temperature given sulfur at each pressure 14, 21, 23, 30, 40
    '''
    ## Low S side of eutectic
    def T_p3l(self,S):
            return self.p3lsp(S)

    def T_p10l(self,S):
            return self.p10lsp(S)

    def T_p14l(self,S):
            return self.p14lsp(S)

    def T_p21l(self,S):
            return self.p21lsp(S)

    def T_p23l(self,S):
            return self.p23lsp(S)

    def T_p30l(self,S):
            return self.p30lsp(S)

    def T_p40l(self,S):
            return self.p40lsp(S)

    ## High S side of eutectic
    def T_p14h(self,S):
            if S<21.:
                    return self.p14msp(S)
            else:
                    return self.p14hsp(S)

    def T_p21h(self,S):
            if S<25.:
                    return self.p21hsp1(S)
            else:
                    return self.p21hsp2(S)

    def T_p23h(self,S):
            if S<21.:
                    return self.p23hsp1(S)
            else:
                    return self.p23hsp2(S)

    def T_p30h(self,S):
            return self.p30hsp(S)

    def T_p3h(self,S):
            return self.p3hsp(S)

    def T_p10h(self,S):
            return self.p10hsp(S)

    def T_p40h(self,S):
            return self.p40hsp(S)

    def T_SPl(self,S,P):
            Pfit = [3.,10.,14.,21.,23.,30.,40.]
            Tfit = [self.T_p3l(S),self.T_p10l(S),self.T_p14l(S),self.T_p21l(S),self.T_p23l(S),self.T_p30l(S),self.T_p40l(S)]
#               Pfit = [3.,14.,21.,23.,30.,40.]
#               Tfit = [self.T_p3l(S),self.T_p14l(S),self.T_p21l(S),self.T_p23l(S),self.T_p30l(S),self.T_p40l(S)]
            return float(UnivariateSpline(Pfit,Tfit,k=1,s=0)(P))

    def T_SPh(self,S,P):
            Pfit = [3.,10.,14.,21.,23.,30.,40.]
            Tfit = [self.T_p3h(S),self.T_p10h(S),self.T_p14h(S),self.T_p21h(S),self.T_p23h(S),self.T_p30h(S),self.T_p40h(S)]
#               Pfit = [3.,14.,21.,23.,30.,40.]
#               Tfit = [self.T_p3h(S),self.T_p14h(S),self.T_p21h(S),self.T_p23h(S),self.T_p30h(S),self.T_p40h(S)]
            return float(UnivariateSpline(Pfit,Tfit,k=1,s=0)(P))

    def T_SP(self,S,P):
        '''
        Finds solidus temperature, given a sulfur wt% and pressure
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
        '''
        P = P/(1.e9)
        S = S*100.
        assert(S>=0. and S<=35.)
        assert(P>=0. and P<=45.)
        if S<10.:
                return self.T_SPl(S,P)
#               elif S>18.27:
        elif S>30.:
                return self.T_SPh(S,P)
        else:
                return max(self.T_SPl(S,P),self.T_SPh(S,P))

    def find_eutectic(self):
        '''
        Find the eutectic from the intersection of T_SPl and T_SPh
        '''
        s_eut = [ ]; t_eut = []
        for p in np.array([3.,10.,14.,21.,23.,30.,40.]):
#             f1 = lambda x: self.T_SPl(x,p)
#             f2 = lambda x: self.T_SPh(x,p)
#             f = lambda x: np.max((self.T_SPl(x,p),self.T_SPh(x,p)))
            f = lambda x: self.T_SP(x/100,p*1.e9)

            s = sp.optimize.fminbound(f,10.,30.)
#             s = findIntersection(f1,f2,15)
            t = f(s)
            s_eut.append(s);t_eut.append(t)
        
        self.S_eutectic = np.array(s_eut)
        self.T_eutectic = np.array(t_eut)

        # linear fits to the eutectic
#         self.eutectic_T_P = UnivariateSpline(self.pressures, self.T_eutectic,k=1,s=0)
        self.eutectic_S_P = UnivariateSpline(self.pressures, self.S_eutectic,k=1,s=0)

        self.eutectic_T_P = sp.interpolate.interp1d(self.pressures,self.T_eutectic)

        self.eutectic_T_S = UnivariateSpline(self.S_eutectic[::-1],\
                self.T_eutectic[::-1],k=1,s=0)
        self.eutectic_P_S = UnivariateSpline(self.S_eutectic[::-1],\
                self.pressures[::-1],k=1,s=0)

    def eutectic(self,p):
        '''
        Return eutectic T and S as a function of P.
        '''

        return self.eutectic_T_P(p/1.e9), self.eutectic_S_P(p/1.e9)/100.

    def eutectic_S(self,s):
        '''
        Return eutectic P and T as a function of S.
        '''

        return self.eutectic_P_S(s*100.)*1.e9, self.eutectic_T_S(s*100.)

    def check_solid(self,S,P,T):
        '''
        checks whether point in S,P,T space is above or below solidus
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
                T: temperature (K)
        '''
        return bool(self.T_SP(S,P) > T)

    def dTdP_SP(self,S,P,h=.005*1.e9):
        '''
        Estimates the clapeyron slope at a givin pressure and composition
        for comparison. Uses a central finite difference method.
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
        '''
        return (self.T_SP(S,P+h) - self.T_SP(S,P-h))/(2.*h)

    def is_Fe_rich(self,S,P,dS=.001):
        '''
        Returns True if on the Fe-rich side of the eutectic.
        '''
        if self.T_SP(S+dS,P) < self.T_SP(S,P):
            return True
        else:
            return False

class Solver_no14(Solver):
    '''
    Version of the Solver class which omits data from Chen at 14 GPa. (So clapeyron
    slopes are monatonic at low S concentrations.
    '''

    def T_SPl(self,S,P):
            Pfit = [3.,10.,21.,23.,30.,40.]
            Tfit = [self.T_p3l(S),self.T_p10l(S),self.T_p21l(S),self.T_p23l(S),self.T_p30l(S),self.T_p40l(S)]
            return float(UnivariateSpline(Pfit,Tfit,k=1,s=0)(P))

    def T_SPh(self,S,P):
            Pfit = [3.,10.,21.,23.,30.,40.]
            Tfit = [self.T_p3h(S),self.T_p10h(S),self.T_p21h(S),self.T_p23h(S),self.T_p30h(S),self.T_p40h(S)]
            return float(UnivariateSpline(Pfit,Tfit,k=1,s=0)(P))

class Dumberry_liquidus(object):
    def __init(self):
        pass
    def eutectic(self,p):
        w = 0.11 + 0.187 * np.exp ( -0.065 * p / 1.e9 )
        if p < 14.e9:
            t = 1265. - 11.15 * ( p/1.e9 - 3.) 
        elif p < 21.e9:
            t = 1143. + 29. * ( p/1.e9 - 14.)
        else:
            t = 1346. + 13. * (p/1.e9 - 21.)
        return t,w

    def iron_melting_curve(self,p):
        return Tm_anzellini(p)

    def T_SP(self,w,p):
        '''
        Finds solidus temperature, given a sulfur wt% and pressure
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
        '''

        T0 = self.iron_melting_curve(p)
        Teut,weut = self.eutectic(p)
        n=100

        return T0 - (T0 - Teut)/weut*w

    def check_solid(self,S,P,T):
        '''
        checks whether point in S,P,T space is above or below solidus
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
                T: temperature (K)
        '''
        return bool(self.T_SP(S,P) > T)

    def dTdP_SP(self,S,P,h=.005*1.e9):
        '''
        Estimates the clapeyron slope at a givin pressure and composition
        for comparison. Uses a central finite difference method.
        args:
                S: sulfur weight percent (0.0 to 1.0)
                P: pressure (Pa)
        '''
        return (self.T_SP(S,P+h) - self.T_SP(S,P-h))/(2.*h)

    def is_Fe_rich(self,S,P):
        '''
        Returns True if on the Fe-rich side of the eutectic.
        '''
        if S < self.eutectic(P)[1]:
            return True
        else:
            return False


def makeFigure(sol,fname,eutectic_cutoff=True):
    Sinterp = np.linspace(0.,35.0,100)/100.
    # Pinterp = np.array([14.,20.,21.,23.,27.,30.,33.,37.,40.])*1.e9
    Pall = np.linspace(3.,40.,20)*1.e9

    Porig = np.array([3.,10.,14.,21.,23.,30.,40.])*1.e9
#     Porig = np.array([3.,10.,14.,21.,23.,40.])*1.e9

#     sol = solver()

    f = plt.figure()
    ax = plt.subplot(111)

    for P in Pall:
            Ttemp = []
            Teut, Seut = sol.eutectic(P)
            for S in Sinterp:
                    Ttemp.append(sol.T_SP(S,P))
            Ttemp = np.array(Ttemp)
            if eutectic_cutoff:
                ax.plot(Sinterp[Sinterp < Seut],Ttemp[Sinterp < Seut],'k--')
            else:
                ax.plot(Sinterp,Ttemp,'k--')

    eutectic = []
    for P in Porig:
            Ttemp = []
            Teut, Seut = sol.eutectic(P)
            eutectic.append([Seut,Teut])
            for S in Sinterp:
                    Ttemp.append(sol.T_SP(S,P))
            Ttemp = np.array(Ttemp)
            if eutectic_cutoff:
                ax.plot(Sinterp[Sinterp < Seut],Ttemp[Sinterp < Seut],'-',lw=2,label=str(P/1.e9)+' GPa')
            else:
                ax.plot(Sinterp,Ttemp,'-',lw=2,label=str(P/1.e9)+' GPa')
    eutectic = np.array(eutectic)



#     Teut = sol.eutectic_T_P(Porig/1.e9)
#     Seut = sol.eutectic_S_P(Porig/1.e9)
    ax.plot(eutectic[:,0],eutectic[:,1],'-',color='0.5',lw=2,label='Eutectic')

    plt.xlabel('Sulfur (wt. frac)')
    plt.ylabel('Temperature (K)')
#     plt.title('Fe-FeS Liquidus')
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.show()
    plt.savefig(fname)
    return sol

if __name__ == "__main__":

    fig_size = [800/72.27 ,700/72.27]
    params = {'backend': 'ps', 'axes.labelsize': 28, 'text.fontsize': 28, 'legend.fontsize': 22,
              'xtick.labelsize': 20, 'ytick.labelsize': 20, 
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
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('font',family='sans-serif')

    sol = Solver_no14()
#     sol = Solver()
    makeFigure(sol,'model_liquidus14.png',eutectic_cutoff=False)

    plt.show()
    # Check that the pure iron end-member is consistent
    f2 = plt.figure
    ax2 = plt.subplot(111)
    P = np.linspace(3.,45.,100)*1.e9
    Tm_fit = np.array([sol.T_SP(0.01,p) for p in P])
    Tm_anz = Tm_anzellini(P)

    ax2.plot(P,Tm_fit)
    ax2.plot(P,Tm_anz)

    plt.show()
