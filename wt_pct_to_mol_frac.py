'''
Quick script to covert between wt % to mol fraction for metal alloys
'''
import numpy as np

mFe = 55.845
mSi = 28.0855
mS = 32.066
mC = 12.011
mO = 15.9994


wC = .0021
wS = .0797
wSi = .0540
wO = .0011
wFe = 1. - wSi - wS - wC

m = np.array([mC,mFe,mS,wO,mSi])
w = np.array([wC,wFe,wS,wO,wSi])

x_melt = w / m / np.sum( w / m )

print x_melt


wC  =  .0007
wS  =  .0006
wSi =  .1065
wO  =  .0023
wFe = 1. - wSi - wS - wC

m = np.array([mC,mFe,mS,wO,mSi])
w = np.array([wC,wFe,wS,wO,wSi])

x_solid = w / m / np.sum( w / m )

print x_solid
