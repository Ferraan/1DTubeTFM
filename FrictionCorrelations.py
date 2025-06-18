# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:21:08 2025

@author: blas_fe

This file contains all the friction correlations needed to
compute the pressure drop and heat transfer. 
"""
import math
import numpy as np

# Equivalence between Ra and ks, Stimpson 2017 ASME, eq 9
def RaKsStimpson(Ra_Dh):
    if(Ra_Dh>0.028):
        return (18 * Ra_Dh - 0.05)
    else:
        return (0)
    
# Zigrang and Sylvester friction factor from Heat transfer Anthony Mills pag367 eq 4.138
def frictionFactor(Re, ks_R):
    laminar = 64 / Re # Laminar regime "independent of roughness"
                     # Not independant when relative roughness > 7%
                     # See Stimpson 2016 ASME

    turbulent = ((-2 * np.log10(ks_R / 7.4 - 5.02 / Re * np.log10( ks_R / 7.4 + 13 / Re)))**(-2))
    return np.where(Re < 2300, laminar, turbulent)

# Returns the pressure downstream of a tube with friction factor f
# DeltaP = f * (L*V*V*rho)/(2*D)
# p2 - p1 = f * (L*V*V*rho)/(2*D)
# p2 = p1 + f * (L*V*V*rho)/(2*D)
def DarcyWeisbach(f,p,L,D,density,v):
    return(p - f * (L * density * v**2 ) / (2 * D))
# Vectorize to compute the pressure loss of all the channel at the same time
#DWvec = np.vectorize(DarcyWeisbach)