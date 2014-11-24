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
    lats = np.array([85., 45., 0., -45., -85. ] )
#    lons = np.array([0.,90.,180.,270.])
#    temperature = np.array([  175., 175., 175., 175., \
#                             400., 310., 400., 310.,\
#                             430., 330., 430., 330.,\
#                             400., 310., 400., 310.,\

#                             175., 175., 175., 175.])
#    temperature = np.asarray([[  175., 175., 175., 175.], \
#                           [  400., 310., 400., 310.],\
#                           [  430., 330., 430., 330.],\
#                           [  400., 310., 400., 310.],\
#                           [  175., 175., 175., 175.]])
    lats = np.pi/2. - lats *np.pi/180.
    lons = lons * np.pi/180.
    points = np.vstack([lats, lons])
    func = interp.RectSphereBivariateSpline( lats,lons, temperature)
    
    x = 0.
    if theta < np.pi/2.:
        x = np.sqrt( theta/(np.pi/2.))*200. + 150. +100.*np.cos(2.*phi)    
    elif theta >= np.pi/2.:
        x =  np.sqrt( (np.pi-theta)/(np.pi/2.))*200. + 150. +100.*np.cos(2.*phi)    
 
    return x

    return func(theta, phi)


def pattern( theta, phi ):
    x = np.cos(theta)*10. + np.sin(phi)*5
#    x = spherical_harmonics.real_spherical_harmonic(theta, phi, 3, 1 )
    return x

coeffs = spherical_harmonics.spherical_harmonic_transform( surface_temperature, 5 )
#coeffs = [ np.asarray(c)/2. for c in coeffs]
print coeffs
spherical_harmonics.plot_spherical_harmonic_expansion( coeffs, surface_temperature)    
                       
