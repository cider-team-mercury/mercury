import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

def real_spherical_harmonic( theta, phi, l, m):
    assert( m <= l )
    assert( l >= 0. )

    if m==0:
        val = sph_harm(m, l, phi, theta)
        return val.real
    elif m < 0:
        val = 1.0j/np.sqrt(2.) * (sph_harm(m,l,phi, theta) 
              - np.power(-1., m ) * sph_harm(-m, l, phi, theta) )
        return val.real
    elif m > 0:
        val = 1.0/np.sqrt(2.) * (sph_harm(-m,l,phi, theta) 
              + np.power(-1., m ) * sph_harm(m, l, phi, theta) )
        return val.real

def spherical_harmonic_transform( func, lmax ):
    coeffs = []
    glpoints, glweights = leggauss(lmax)
    glpoints = np.arccos(glpoints)
    fpoints = np.linspace(0.,np.pi*2., lmax+1, endpoint=False)
    for l in range(lmax+1):
        c = []
        for m in range(2*l+1):
            mp = m-l
            val = 0.
            
            for theta, weight in zip(glpoints, glweights):
                for phi in fpoints:
                    val += weight*real_spherical_harmonic( theta, phi, l, mp)*func(theta,phi)
            if val < 1.e-13: 
                val = 0.
            c.append(val)
        coeffs.append(c)
    return coeffs


def plot_spherical_harmonic_expansion( coeffs, func = None):
    
    res = 40
    lats = np.linspace(0.,np.pi, res)
    lons = np.linspace(0.,2.*np.pi, 2*res)
    LONS, LATS = np.meshgrid(lons, lats)
    T = np.empty_like(LATS)
    T2 = np.empty_like(LATS)
    
  
    for l,c in enumerate(coeffs):
        assert(len(c) == 2*l+1)
        for m,val in enumerate(c):
            mp = m-l
            for i,theta in enumerate(lats):
                for j,phi in enumerate(lons):
                    T[i,j] += val*real_spherical_harmonic(theta, phi, l, mp)
                    if(func != None):
                        T2[i,j] = func(theta, phi)

    map = Basemap( projection='hammer', lat_0=30., lon_0=0.)
    x,y=map(180.-LONS*180./np.pi, 90.-LATS*180./np.pi)

    minval = np.min(T)
    maxval = np.max(T)
    print "Minimum: ", minval, "Maximum: ", maxval, "Variation: ", (maxval-minval)/minval

    if( func == None):
        map.pcolor(x, y, T, cmap='Spectral_r')
        cb1 = plt.colorbar()
        plt.show()
    else:
        plt.subplot(121)
        map.pcolor(x, y, T)
        cb1 = plt.colorbar()
        plt.subplot(122)
        map.pcolor(x, y, T2)
        cb2 = plt.colorbar()
        plt.show()
