# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:25:13 2024

@author: 44788
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define spatial step and domain.
x_0 = 0                         # x_0 : Left boundary x
x_1 = 1                        # x_1 : Right boundary x
dx = 0.01                       # dx: Spatial step.
Dx = int((x_1 - x_0 + dx) / dx) # Dx: Number of spatial steps.
x = np.linspace(x_0, x_1, Dx)   # x : Spatial mesh.

# Define time step and domain.
t_0 = 0                         # t_0 : Initial time
t_1 = 100                         # t_1 : Final time
dt = 0.01                       # dt: Time step.
Dt = int((t_1 - t_0 + dt) / dt) # Dt: Number of time steps.
t = np.linspace(t_0, t_1, Dt)   # t : Time mesh.

# Define parameter values.
D_phi_m = 0.01                  # Estimation from paper
D_phi_c = 0.01                  # Estimation from paper
Pe = 1                          # UNKNOWN
N_R7 = 30000                    # From paper
alpha_1 = 1e-7                  # From paper
alpha_2 = 1                     # UNKNOWN
beta_p = 4.6*1e-6               # From paper
beta_m = 12.5                   # From paper
chi_u = 1e-10                       # UNKNOWN
        

# Define initial conditions
c_0 = (1-x**(1/2))
phi_m_0 = 0.1 * np.sin(x * np.pi / (x_1 - x_0))
phi_c_0 = 0.2 * np.sin(x * np.pi / (x_1 - x_0))  # New initial condition for c

# Define PDE as a System of ODEs
def pde(y, t):
    c = y[:Dx]
    phi_m = y[Dx:2*Dx]
    phi_c = y[2*Dx:]
    
    dc_dt = np.zeros_like(c)
    dphi_m_dt = np.zeros_like(phi_m)
    dphi_c_dt = np.zeros_like(phi_c)
    
    dc_dt[1:-1] = 1/dx * ((c[:-2]- 2*c[1:-1] + c[2:])/dx - Pe*c[1:-1]) - \
                        N_R7*beta_p*c[1:-1]*phi_m[1:-1] + \
                        alpha_1*beta_m*phi_c[1:-1] - \
                        alpha_2*c[1:-1]
    
    dphi_m_dt[1:-1] = 1/dx * D_phi_m*(phi_m[:-2] -2*phi_m[1:-1] + phi_m[2:]) / dx - \
                        1/alpha_1*beta_p*c[1:-1]*phi_m[1:-1] + \
                        beta_m*phi_c[1:-1]
    
    
    dphi_c_dt[1:-1] = 1/dx * D_phi_c*((phi_c[:-2] - 2*phi_c[1:-1]+phi_c[2:]) / dx - \
                        chi_u*phi_c[1:-1]*((c[:-2] - c[2:])/(2*dx))) + \
                        1/alpha_1*beta_p*c[1:-1]*phi_m[1:-1] - \
                        beta_m*phi_c[1:-1]
    
    # Boundary Conditons
    # Dirichlet (x,c) = (0,1),(1,0)
    # No-flux Boundaries
    
    #phi_m[0] = phi_m[1]
    
     
    return np.concatenate([dc_dt, dphi_m_dt, dphi_c_dt])

# Solve ODE
initial_conditions = np.concatenate([c_0, phi_m_0, phi_c_0])
solution_vec = odeint(pde, initial_conditions, t)

# Plot results
for i, t_i in enumerate(t):
    if i % 100 == 0:
        plt.figure(figsize=(7, 6), dpi=200)
        plt.plot(x, solution_vec[i, :Dx], label='c', color='darkblue')
        plt.plot(x, solution_vec[i, Dx:2*Dx], label=r'$\phi_m$', color='darkred')
        plt.plot(x, solution_vec[i, 2*Dx:], label=r'$\phi_c$', color='darkgreen')  # Plot for c
        plt.xlabel('x')
        plt.xlim(x_0, x_1)
        plt.ylim(-0.1, 1.1)
        plt.title(f"t={t_i:.3f}")
        plt.legend()
        plt.grid()
        plt.show()
