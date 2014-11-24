import numpy as np
import matplotlib.pyplot as plt
import spherical_harmonics


#Spherical harmonic coefficients from Aharonson et al., 2004,
#which are themselves pseudo-fits to the thermal model of 
#Vasavada et. al. 1999.  The reported values go up to l=4,
#but the plots look like they are truncated at l=2.  I cannot
#think of a way to justify an l=4 temperature structure given
#the symmetries, so I too will truncate at l=2
surface_temperature_coeffs = [ [1230., ],     #l=0
           [0., 0., 0.], #l=1
           [0., 0., -131., 0., 81.], #l=2
           [0., 0., 0., 0., 0., 0., 0.], #l=3
           [0., 0., 0., 0., -82., 0., 26., 0., 0.] ]#l=4 
surface_temperature_coeffs = [np.array(c) for c in surface_temperature_coeffs]

k = 4. #Thermal conductivity of mafic rocks
outer_radius = 2440.e3
inner_radius = 2020.e3
T_inner = 1300. #Inner temperature



Y_00 = 0.5 * np.sqrt(1./np.pi)
T_outer = surface_temperature_coeffs[0][0]*Y_00  # Average outer temperature
eta = inner_radius/outer_radius
factors = np.zeros( len(surface_temperature_coeffs) )

for l,f in enumerate(factors):
    factors[l] =( (2.*l+1) * np.power(eta, (l-1.) ) )\
                / ( 1. - np.power(eta, (2.*l+1.) ) )
    factors[l] = -k*factors[l]/outer_radius

flux_coeffs = [ factors[l]*surface_temperature_coeffs[l] for l in range(len(surface_temperature_coeffs) )]
q_average = k * (T_inner-T_outer)/inner_radius * (1./(1.-eta) )
flux_coeffs[0][0] = q_average/Y_00
print flux_coeffs

print "Total heat flux : ", 4.*np.pi*inner_radius*inner_radius*q_average/1.e12, "TW"

spherical_harmonics.plot_spherical_harmonic_expansion( flux_coeffs)    
