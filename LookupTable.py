# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:54:09 2025

@author: blas_fe
"""
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def Create_lookup_table():
    import cantera as ct
    CanteraAir = ct.Solution('air.yaml')
    temperatures = np.arange(225, 1000, 4)  # Example temperatures in Kelvin
    pressures = np.arange(0.5e+5, 10e+5, 1e+4)  # Example pressures in Pascals
    
    # Create 2D arrays to store the cp and viscosity values for each combination of pressure and temperature
    cp_values = np.zeros((len(temperatures), len(pressures)))
    viscosity_values = np.zeros((len(temperatures), len(pressures)))
    density_values = np.zeros((len(temperatures), len(pressures)))
    cv_values = np.zeros((len(temperatures), len(pressures)))
    
    # Populate the cp and viscosity arrays using the get_thermo_properties function
    for i, pressure in enumerate(pressures):
        for j, temperature in enumerate(temperatures):
            CanteraAir.TP = temperature, pressure
            cp, viscosity, rho, cv= CanteraAir.cp, CanteraAir.viscosity, CanteraAir.density, CanteraAir.cv
            cp_values[j, i] = cp  # Temperature index is row (j), pressure index is column (i)
            viscosity_values[j, i] = viscosity
            density_values[j, i] = rho
            cv_values[j, i] = cv
    
    # Create the interpolator for cp and viscosity values
    cp_interpolator = RegularGridInterpolator((temperatures, pressures), cp_values, method='linear', bounds_error=True, fill_value=None)
    viscosity_interpolator = RegularGridInterpolator((temperatures, pressures), viscosity_values, method='linear', bounds_error=True, fill_value=None)
    density_interpolator = RegularGridInterpolator((temperatures, pressures), density_values, method='linear', bounds_error=True, fill_value=None)
    cv_interpolator = RegularGridInterpolator((temperatures, pressures), cv_values, method='linear', bounds_error=True, fill_value=None)
    interpolators = [cp_interpolator, viscosity_interpolator, density_interpolator, cv_interpolator]
    return interpolators

# Interpolation function using RegularGridInterpolator
def pT_interp(interpolators, pressure, temperature):
    cp_interpolator = interpolators[0]
    viscosity_interpolator = interpolators[1] 
    density_interpolator = interpolators[2] 
    cv_interpolator = interpolators[3] 
    # Interpolate the values for the given pressure and temperature
    cp_interpolated = cp_interpolator((temperature, pressure))  # Note the order: (temperature, pressure)
    viscosity_interpolated = viscosity_interpolator((temperature, pressure))
    cv_interpolated = cv_interpolator((temperature, pressure))  # Note the order: (temperature, pressure)
    density_interpolated = density_interpolator((temperature, pressure))  # Note the order: (temperature, pressure)
    gamma_interpolated = cp_interpolated / cv_interpolated
    return cp_interpolated, cv_interpolated, viscosity_interpolated, density_interpolated, gamma_interpolated