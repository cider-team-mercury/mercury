import numpy as np
import scipy.interpolate as interp
import spherical_harmonics
import matplotlib.pyplot as plt


def surface_temperature(theta, phi):
    """
    Visually reading points off of Vasavada et al, 1999
    """
    lats = np.array( [-85., -85., -85., -85., \
                       -45., -45., -45., -45, \
                       0., 0., 0., 0.,\
                       45., 45., 45., 45, \
                       85., 85., 85., 85.])
    lons = np.array( [ 0., 90., 180., 270.,\
                       0., 90., 180., 270.,\
                       0., 90., 180., 270.,\
                       0., 90., 180., 270.,\
                       0., 90., 180., 270.])
#    lats = np.array([85., 45., 0., -45., -85. ] )
#    lons = np.array([0.,90.,180.,270.])
    temperature = np.array([  175., 175., 175., 175., \
                             400., 310., 400., 310.,\
                             430., 330., 430., 330.,\
                             400., 310., 400., 310.,\
                             175., 175., 175., 175.])
#    temperature = np.asarray([[  175., 175., 175., 175.], \
#                           [  400., 310., 400., 310.],\
#                           [  430., 330., 430., 330.],\
#                           [  400., 310., 400., 310.],\
#                           [  175., 175., 175., 175.]])
    lats = np.pi/2. - lats *np.pi/180.
    lons = lons * np.pi/180.
    points = np.vstack([lats, lons])
#    func = interp.RectSphereBivariateSpline( lats,lons, temperature)
    func = interp.NearestNDInterpolator( points.T, temperature)
    
    x = 0.
    if theta < np.pi/2.:
        x = np.sqrt( theta/(np.pi/2.))*200. + 150. +100.*np.cos(2.*phi)    
    elif theta >= np.pi/2.:
        x =  np.sqrt( (np.pi-theta)/(np.pi/2.))*200. + 150. +100.*np.cos(2.*phi)    
 
    print theta, phi, x
    return x

    return func(theta, phi)


def pattern( theta, phi ):
    x = np.cos(theta)*10.
    print x
    return x

coeffs = spherical_harmonics.spherical_harmonic_transform( pattern, 2 )
spherical_harmonics.plot_spherical_harmonic_expansion( coeffs)    
                       
