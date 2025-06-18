# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:59:32 2025

@author: blas_fe
"""
import math
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt

# Geometry
L = 0.06  # m
D = 1e-3  # m
Ac = np.pi * (D/2)**2

# Numerical parameters
N = 7
epsilon = 1e-6

# Boundary conditions
Tin = 5.38e+01+273.15  # K
Pin = 5.38e+00*1e+5  # Pa
mflowin = 2.87e-01/1000  # kg/s

Tout = 2.95e+02+273.15  # K
Pout = ct.one_atm  # Pa, 1atm

# Create air object to obtain thermophysical and transport properties
CanteraAir = ct.Solution('air.yaml')
Rair = 287.052874 # J kg−1 K−1
# Begin discretization
COORD = np.linspace(0, L, N)
DeltaX = COORD[1]-COORD[0]

# Wall temperature
# Linear profile between T1 and T2
def T_lin(x, T1, T2):
    return (T2-T1)/L*x+T1

# Linear profile between Pin and Pout
def P_lin(x, P1, P2):
    return (P2 - P1) / L * x + P1

# Equivalence between Ra and ks, Stimpson 2017 ASME, eq 9
def RaKsStimpson(Ra_Dh):
    if(Ra_Dh>0.028):
        return (18 * Ra_Dh - 0.05)
    else:
        return (0)
    
# Zigrang and Sylvester friction factor from Heat transfer Anthony Mills pag367 eq 4.138
def frictionFactor(Re, ks_R):
    if(Re < 2300): #Laminar regime "independent of roughness"
        return (64 / Re)
    else:
        return ((-2 * math.log10(ks_R / 7.4 - 5.02 / Re * math.log10( ks_R / 7.4 + 13 / Re)))**(-2))
# Vectorize it to increase speed
ff = np.vectorize(frictionFactor)

# Returns the pressure downstream of a tube with friction factor f
def DarcyWeisbach(f,p,L,D,density,v):
    return(p - f * (L / D) * (density * v**2 ) / 2)
DWvec = np.vectorize(DarcyWeisbach)

## COMPUTATIONS
# %%
Ra = 12.51e-6 # m CHANGE TO CHANGE FRICTION FACTOR OF CHANNEL
KsDh = RaKsStimpson(Ra/D)
KsR = KsDh*2 
# Initialize array
init_var = np.zeros((N, 1))
T = np.zeros_like(init_var)
p = np.zeros_like(init_var)
rho = np.zeros_like(init_var)
u = np.zeros_like(init_var)
mu = np.zeros_like(init_var)
cp = np.zeros_like(init_var)
cv = np.zeros_like(init_var)
gamma = np.zeros_like(init_var)

T[0, 0] = Tin
T[-1, 0] = Tout
p[0, 0] = Pin
p[-1, 0] = Pout

# First step, assume linear profile
T[1:-1, 0] = T_lin(COORD[1: -1], Tin, Tout)
p[1:-1, 0] = P_lin(COORD[1: -1], Pin, Pout)

## Index 0
CanteraAir.TP = T[0, 0], p[0, 0]
rho[0] = CanteraAir.density
mu[0] = CanteraAir.viscosity
cp[0] = CanteraAir.cp
cv[0] = CanteraAir.cv
gamma[0] = cp[0] / cv[0]
# Compute velocity from conservation equation
u[0] = mflowin / (Ac * rho[0])

CanteraAir.TP = T[-1, 0], p[-1, 0]
rho[-1] = CanteraAir.density
mu[-1] = CanteraAir.viscosity
cp[-1] = CanteraAir.cp
cv[-1] = CanteraAir.cv
gamma[-1] = cp[-1] / cv[-1]
# Compute velocity from conservation equation
u[-1] = mflowin / (Ac * rho[-1])

## Index N
for i in range(1, N-1):
    # Get density at this T and P
    CanteraAir.TP = T[i][-1], p[i][-1]
    rho[i, -1] = CanteraAir.density
    # Get viscosity to compute Reynolds
    mu[i, -1] = CanteraAir.viscosity
    # Get cp and cv
    cp[i, -1] = CanteraAir.cp
    cv[i, -1] = CanteraAir.cv
    # Get ratio of specific heats
    gamma[i, -1] = cp[i, -1] / cv[i, -1]
    # Compute velocity from conservation equation
    u[i,-1] = mflowin/(Ac*rho[i,-1])
    
Reynolds = u[:, -1]*D*rho[:, -1]/mu[:, -1]
# Iterative solver
error = 1
count = 0
while error > epsilon:
    count += 1
    # Append new iteration onto the variables
    T = np.append(T, T[:, -1][:, np.newaxis], axis = 1)
    
    #Update pressure with Darcy-Weisbach eq
    #FERRAN recalculate
    # Compute friction factor with the previous iteration Reynolds (f,p,L,D,density,v)
    friction_factor_vector = ff(Reynolds[:], KsR)
    # Compute new pressures with DarcyWeisbach
    new_p = DWvec(friction_factor_vector[0: -1], p[0: -1, -1], DeltaX, D, rho[0: -1, -1], u[0: -1, -1])
    new_p = np.append(p[0, -1], new_p).reshape(-1,1)
    # Append the new pressures
    p = np.append(p, new_p, axis = 1)
    
    # Update rho, u and mu
    rho = np.append(rho, np.zeros_like(init_var), axis = 1)
    u = np.append(u, np.zeros_like(init_var), axis = 1)
    mu = np.append(mu, np.zeros_like(init_var), axis = 1)
    
    rho[0, -1] = rho[0, -2]
    u[0, -1] = u[0, -2]
    mu[0, -1] = mu[0, -2]
    # 
    for i in range(1, N):
        # Get density at this T and P
        CanteraAir.TP = T[i][-1], p[i][-1]
        rho[i,-1] = CanteraAir.density
        # Get viscosity to compute Reynolds
        mu[i,-1] = CanteraAir.viscosity
        # Get cp and cv
        cp[i, -1] = CanteraAir.cp
        cv[i, -1] = CanteraAir.cv
        # Get ratio of specific heats
        gamma[i, -1] = cp[i, -1] / cv[i, -1]
        # Compute velocity from conservation equation
        u[i, -1] = mflowin / (Ac * rho[i, -1])
    error = np.sum(np.abs(u[:, -1] - u[:, -2]))
    
    if count > 1000:
        break

Reynolds = u[:, -1]*D*rho[:, -1]/mu[:, -1]
Mach = u[:, -1]/np.sqrt(gamma[:, -1]*Rair*T[:, -1])




## PLOTTING
# %%

fig, axs = plt.subplots(3, 2, figsize=(12, 8))

axs[0, 0].plot(COORD/L, u[:,-1])
axs[0, 0].set_ylabel('u [m/s]')
axs[0, 0].grid(True)

axs[0, 1].plot(COORD/L, T[:,-1])
axs[0, 1].set_ylabel('Temperature [K]')
axs[0, 1].grid(True)

axs[1, 0].plot(COORD/L, p[:,-1])
axs[1, 0].set_ylabel('Pressure [Pa]')
axs[1, 0].grid(True)

axs[1, 1].plot(COORD/L, rho[:,-1])
axs[1, 1].set_ylabel('Density [kg/m³]')
axs[1, 1].grid(True)

axs[2, 0].plot(COORD/L, Reynolds)
axs[2, 0].set_xlabel('x/L')
axs[2, 0].set_ylabel('Reynolds')
axs[2, 0].grid(True)

axs[2, 1].plot(COORD/L, Mach)
axs[2, 1].set_xlabel('x/L')
axs[2, 1].set_ylabel('Mach')
axs[2, 1].grid(True)


plt.tight_layout()
plt.show()