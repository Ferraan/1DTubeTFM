# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:13:51 2025

@author: blas_fe
"""
import matplotlib.pyplot as plt
def Plot(Air, COORD, L):
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    
    axs[0, 0].plot(COORD/L, Air.u[:,-1])
    axs[0, 0].set_ylabel('u [m/s]')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(COORD/L, Air.T[:,-1])
    axs[0, 1].set_ylabel('Temperature [K]')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(COORD/L, Air.p[:,-1])
    axs[1, 0].set_ylabel('Pressure [Pa]')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(COORD/L, Air.rho[:,-1])
    axs[1, 1].set_ylabel('Density [kg/m³]')
    axs[1, 1].grid(True)
    
    axs[2, 0].plot(COORD/L, Air.Reynolds)
    axs[2, 0].set_xlabel('x/L')
    axs[2, 0].set_ylabel('Reynolds')
    axs[2, 0].grid(True)
    
    axs[2, 1].plot(COORD/L, Air.Mach)
    axs[2, 1].set_xlabel('x/L')
    axs[2, 1].set_ylabel('Mach')
    axs[2, 1].grid(True)
    
    
    plt.tight_layout()
    plt.show()