'''
core_partition.py

Contains functions for dealing with the partitioning of light elements
between a growing solid inner core and liquid outer core, and the 
latent heat released by inner core growth.
'''

import numpy as np
from scipy import integrate
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

# from mercury_minerals import ironSilicideAlloy,ironSulfideSilicideLiquid

# molar masses
from mercury_reference import mFe,mSi,mS,LFe

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

# iron latent heat of fusion
iron_latent_heat = lambda P, T, w : LFe

def partition(w_total1,D1,f_solid):
    '''
    Finds the average composition of a solidifying inner core and residual
    outer core coining  light elements with a total weight percent
    w_total and Distribution coefficients D, when the a fraction of the mass
    of the core f_solid. The calculation is dones by freezing the liquid in 
    number of equal mass batches given by steps.

    Note: if a D = 0, values of f_solid exceeding 1 - w_i for that species are
    nonsensical
    '''

    w_total = np.array(w_total1)
    D = np.array(D1)

    assert len(w_total) == len(D)
    assert f_solid >= 0. and f_solid <= 1. 

    # avoid divide by zeros for trivial case of f_solid = 0.
    if f_solid == 0.:
        return w_total, np.zeros_like(w_total)

    # initial mass (normalized to 1)
    mliq0 = 1. * np.hstack((w_total,1.-np.sum(w_total)))

    # unit mass transfered, based on partition coefficients
    w= lambda m: m / np.sum(m)
    wbatch = lambda mliq, m : - np.hstack( (w(mliq)[:-1]*D, 1. - np.sum(w(mliq)[:-1]*D) ) )

    # integrate to a core mass fraction
    soln = integrate.odeint(wbatch,mliq0,[0.,f_solid])
    mliq = soln[-1]
    msol = mliq0 - mliq

    # check that mtotal stays at ~1 
    mtotal = sum(mliq)+sum(msol)

    # throw an error if f_solid is unreasonable
    for x in np.hstack((msol,mliq)):
            assert x >= 0., "Unreasonable f_solid for partitioning model"

    # return the weight % of the
    return w(mliq)[:-1], w(msol)[:-1]


def density_coexist(w_liquid1,D1,P,T,mat_solid, \
        mat_liquid):
    '''
    Apply partitioning rule to find the composition of a solid coexisting with the
    liquid.
    '''

    w_liquid = np.array(w_liquid1)
    D = np.array(D1)
    assert len(w_liquid) == len(D)+1

    w_coex = np.hstack((D*w_liquid[:-1],1.-np.sum(D*w_liquid[:-1]) ) )

    x_liquid = w_to_x(w_liquid)
    x_coex = w_to_x(w_coex)

    solid = mat_solid(x_coex[1])
    liquid = mat_liquid(x_liquid[0],x_liquid[1])

    solid.set_method('slb3');liquid.set_method('slb3')
    solid.set_state(P,T); liquid.set_state(P,T)

    # return densities
    return solid.density(), liquid.density()

def coeff_comp_expansivity(solution,w_spec,P,T,component=0,wmax=0.2,n=50,\
        method='slb3'):
    '''
    Find the compositional analogue of the coeff. of thermal expansion. (1/wt %)

    1/V * (dV/dw) 
    '''

    w_arr = np.linspace(0.,wmax,50)
    w_eval = w_spec[component]

    assert w_eval >= 0.
    assert w_eval <= wmax
    
    v_list = []
    for w in w_arr:
        if component==0:
            x1 = w_to_x([w,w_spec[1],1.-w_spec[1]-w])[0]
            x2 = w_to_x([w,w_spec[1],1.-w_spec[1]-w])[1]
        elif component==1:
            x1 = w_to_x([w_spec[0],w,1.-w_spec[0]-w])[0]
            x2 = w_to_x([w_spec[0],w,1.-w_spec[0]-w])[1]
        phase = solution(x1,x2)
        phase.set_method(method)
        phase.set_state(P,T)
        v_list.append(phase.V)

    v_arr = np.array(v_list)
    vfunc = UnivariateSpline(w_arr,v_arr)

    return derivative(vfunc,w_eval) / vfunc(w_eval)
