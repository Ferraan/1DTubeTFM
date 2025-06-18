# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:32:09 2025

@author: blas_fe

This file initializes all the fluid variables, creates the arrays and 
assigns the appropriate boundary conditions.
"""
import numpy as np
from LookupTable import pT_interp
from dataclasses import dataclass, field

# Wall temperature
# Linear profile between T1 and T2
def T_lin(x, T1, T2, L):
    return (T2-T1)/L*x+T1

# Linear profile between Pin and Pout
def P_lin(x, P1, P2, L):
    return (P2 - P1) / L * x + P1

def Initialization(N, Ra, D, BCs, Ac, COORD, interpolators, L):
    # Create fluid class with all thermophysical properties as 
    # methods
    @dataclass
    class Fluid:
        N: int
        u: np.ndarray = field(init = False)
        T: np.ndarray = field(init = False)
        p: np.ndarray = field(init = False)
        rho: np.ndarray = field(init = False)
        mu: np.ndarray = field(init = False)
        cp: np.ndarray = field(init = False)
        cv: np.ndarray = field(init = False)
        gamma: np.ndarray = field(init = False)
        Reynolds: np.ndarray = field(init = False)
        Mach: np.ndarray = field(init = False)
        def __post_init__(self):
            self.u = np.zeros((self.N,1))
            self.T = np.zeros((self.N,1))
            self.p = np.zeros((self.N,1))
            self.rho = np.zeros((self.N,1))
            self.mu = np.zeros((self.N,1))
            self.cp = np.zeros((self.N,1))
            self.cv = np.zeros((self.N,1))
            self.gamma = np.zeros((self.N,1))
            self.Reynolds = np.zeros((self.N,1))
            self.Mach = np.zeros((self.N,1))
    # Instantiate the class using N cells for discretization
    Air = Fluid(N = N)  
    
    # Assign boundary conditions
    Air.T[0, 0] = BCs.Tin
    Air.T[-1, 0] = BCs.Tout
    Air.p[0, 0] = BCs.Pin
    Air.p[-1, 0] = BCs.Pout
    
    # First step, assume linear profile
    Air.T[1:-1, 0] = T_lin(COORD[1: -1], BCs.Tin, BCs.Tout, L)
    Air.p[1:-1, 0] = P_lin(COORD[1: -1], BCs.Pin, BCs.Pout, L)
        
    # Get termophysical properties once p and T are known    
    Air.cp[:, -1], Air.cv[:, -1], Air.mu[:, -1], Air.rho[:, -1], Air.gamma[:, -1] = pT_interp(interpolators,
                                  Air.p[:,-1], Air.T[:,-1])
    # Compute velocity from conservation equation
    Air.u[:, -1] = BCs.mflowin / (Ac * Air.rho[:, -1])
    #Compute Reynolds
    Air.Reynolds = Air.u[:, -1] * D * Air.rho[:, -1] / Air.mu[:, -1]
    
    return Air