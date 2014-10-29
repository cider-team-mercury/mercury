'''
core_partition.py
'''

import numpy as np

def partition(w_total1,D1,f_solid,steps=1000):
    '''
    Finds the average composition of a solidifying inner core and residual
    outer core coining  light elements with a total weight percent
    w_total and Distribution coefficients D, when the a fraction of the mass
    of the core f_solid. The calculation is dones by freezing the liquid in 
    number of equal mass batches given by steps.
    '''

    w_total = np.array(w_total1)
    D = np.array(D1)

    assert len(w_total) == len(D)
    assert f_solid >= 0. and f_solid <= 1. 

    # avoid divide by zeros for trivial case of f_solid = 0.
    if f_solid == 0.:
        return w_total, np.zeros_like(w_total)

    dm = 1./float(steps) * f_solid
    
    mliq = 1. * np.hstack((w_total,1.-np.sum(w_total)))
    msol = np.zeros_like(mliq)

    wliq = w_total

    m_in = 0.
    for i in range(steps):
        wbatch = wliq*D

        mbatch = dm* np.hstack((wbatch,1.-np.sum(wbatch)))
        msol += mbatch
        mliq -= mbatch

        # total mass for check
        mtot = np.sum(msol)+np.sum(mliq)

        wliq = mliq[:-1]/np.sum(mliq)
        wsol = msol[:-1]/np.sum(msol)

#         print i, mliq,np.sum(mliq), msol,np.sum(msol), mtot
#         print mbatch, np.sum(mbatch)

    return wliq, wsol
        
        

        

