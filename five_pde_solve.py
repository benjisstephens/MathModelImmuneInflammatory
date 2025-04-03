# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:29:29 2024

@author: Benjamin Stephens
"""

import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# Save images?
toggle_save = True

chemokine_present = True
flow_direction = 'NEG'
steady_state = False

# Define spatial step and domain.
x_0 = 0                              # x_0 : Left boundary x
x_1 = 1                              # x_1 : Right boundary x
dx = 0.01
                                     # dx: Spatial step.
Dx = int((x_1 - x_0 + dx) / dx)      # Dx: Number of spatial steps.
x = np.linspace(x_0, x_1, Dx)        # x : Spatial mesh.

# Define parameter values.
D_c_t =     10                       
D_phi =     0.01                   
D_phi_c =   D_phi                     
D_phi_c_t = D_phi                     
N_R7 =      30000                    
r =         9.8e-7                   
d_c =       1.9e-2                   
d_c_t =     6.4e-3                  
beta_p =    1.5e-4                   
beta_m =    72                     
chi =       4e-3                    
chi_t =     chi                    
gamma =     4e-3 #2.8                      

# %% Get initial conditions from data

# File to be processed
if chemokine_present == False:
   data_file = "M4_wDC_CTRL_POS_pos_export.txt" 
   Pe = 2
   proportion_bound = 0
   cell_count = 78
else:
    if flow_direction == 'NEG':
        data_file = "M12_wDC_CCL21_NEG_pos_export.txt"
        Pe = -2
        proportion_bound = 0.2
        cell_count = 573
    elif flow_direction == 'DIF':
        data_file = "M12_wDC_CCL21_DIF_pos_export.txt"
        Pe = 0
        proportion_bound = 0.36
        cell_count = 143
    elif flow_direction == 'POS':
        data_file = "M12_wDC_CCL21_POS_pos_export.txt"
        Pe = 2
        proportion_bound = 0.36
        cell_count = 250

data = pd.read_csv(data_file)

first_time = data['Time(s)'].min()

first_time_cells = data.groupby('CellID')['Time(s)'].min().eq(first_time).sum()

groups = data.groupby('CellID')

title = data_file.replace('_pos_export.txt', '')

new_x_positions = []  
for cell_id, group in groups:
    time_group = group['Time(s)']

    if first_time < time_group.min() or first_time > time_group.max():
        continue
    
    y_group = group['y(microns)']
    y_pos = np.interp(first_time, time_group, y_group)

    new_x = 1250 - y_pos
    new_x_positions.append(new_x)
    
new_x_positions = np.array(new_x_positions)


kde_1d = gaussian_kde(new_x_positions, bw_method=1 * cell_count**(-1/5))

grid_1d = np.linspace(0, 1200, 200)

density_1d = kde_1d(grid_1d) * first_time_cells


density_rescaled = kde_1d(1200 * x) * first_time_cells
density_interp = interp1d(x, density_rescaled, kind='cubic',
                          bounds_error=False, fill_value="extrapolate")

def phi0(x):
    return density_interp(x) * (1 - proportion_bound) / 0.951

def phic0(x):
    return density_interp(x) * proportion_bound / 0.951

def c_bar(x, Pe, d_c):
    L1 = 0.5 * (Pe + np.sqrt(Pe**2 + 4 * d_c))
    L2 = 0.5 * (Pe - np.sqrt(Pe**2 + 4 * d_c))
    A = 1 / (1 - np.exp(np.sqrt(Pe**2 + 4 * d_c)))
    
    return A*np.exp(L1 * x) + (1-A)*np.exp(L2 * x)

# Define initial conditions
if chemokine_present == True:
    c_0 = c_bar(x, Pe, d_c)
else:
    c_0 = 0*x
c_t_0 = 0*x
phi_0 = phi0(x)
phi_c_0 = phic0(x)
phi_c_t_0 = 0*x

t_0 =   0                                           # t_0 : Initial time
if steady_state == True:                               
    t_1 = 30                                        # t_1 : Final time                                    
else:                                                       
    t_1 = 1800/(1.44e4)                             # dt: Time step.
dt = t_1/100 
Dt =    int((t_1 - t_0 + dt) / dt)                  # Dt: Number of time steps.
t =     np.linspace(t_0, t_1, Dt)                   # t : Time mesh.


# Define PDE as a System of ODEs
def pde(t, y):
    c = y[:Dx]
    phi = y[Dx:2*Dx]
    phi_c = y[2*Dx:3*Dx]
    c_t = y[3*Dx:4*Dx]
    phi_c_t = y[4*Dx:]

    dc_dt = np.zeros_like(c)
    dphi_dt = np.zeros_like(phi)
    dphi_c_dt = np.zeros_like(phi_c)
    dc_t_dt = np.zeros_like(c_t)
    dphi_c_t_dt = np.zeros_like(phi_c_t)

    # Calculate sign of Pe at each step to determine how to do upwind scheme
    Pe_dc_dx = np.zeros(Dx)
    for i in range(1, Dx-1):
        # Backward difference if Pe > 0
        # Forward difference  if Pe < 0
        if Pe > 0:
            Pe_dc_dx[i] = Pe * (c[i] - c[i-1]) / dx
        else:
            Pe_dc_dx[i] = Pe * (c[i+1] - c[i]) / dx

    Pe_dc_t_dx = np.zeros(Dx)
    for i in range(1, Dx-1):
        # Backward difference if Pe > 0
        # Forward difference  if Pe < 0
        if Pe > 0:
            Pe_dc_t_dx[i] = Pe * (c_t[i] - c_t[i-1]) / dx
        else:
            Pe_dc_t_dx[i] = Pe * (c_t[i+1] - c_t[i]) / dx
            
    dphi_c_dx = np.zeros(Dx)
    dc_dx = np.zeros(Dx)
    for i in range(1, Dx-1):
        # Use central difference to approximate dc_dx sign
        dc_dx_test = (c[i+1] - c[i-1]) / (2*dx)
        # Backward difference if chi * dc_dx > 0
        # Forward difference  if chi * dc_dx < 0
        if chi*dc_dx_test > 0:
            dphi_c_dx[i] = (phi_c[i] - phi_c[i-1]) / dx
            dc_dx[i] = (c[i] - c[i-1]) / dx
        else:
            dphi_c_dx[i] = (phi_c[i+1] - phi_c[i]) / dx
            dc_dx[i] = (c[i+1] - c[i]) / dx

    dphi_c_t_dx = np.zeros(Dx)
    dc_t_dx = np.zeros(Dx)
    for i in range(1, Dx-1):
        # Use central difference to approximate dc_dx sign
        dc_t_dx_test = (c_t[i+1] - c_t[i-1]) / (2*dx)
        # Backward difference if chi * dc_dx > 0
        # Forward difference  if chi * dc_dx < 0
        if chi_t*dc_t_dx_test > 0:
            dphi_c_t_dx[i] = (phi_c_t[i] - phi_c_t[i-1]) / dx
            dc_t_dx[i] = (c_t[i] - c_t[i-1]) / dx
        else:
            dphi_c_t_dx[i] = (phi_c_t[i+1] - phi_c_t[i]) / dx
            dc_t_dx[i] = (c_t[i+1] - c_t[i]) / dx

    # dc_dt
    if chemokine_present == True:
        dc_dt[1:-1] = (c[:-2] - 2 * c[1:-1] + c[2:]) / (dx**2) - \
            Pe_dc_dx[1:-1] - \
            N_R7 * beta_p * c[1:-1] * phi[1:-1] + \
            N_R7 * r * beta_m * phi_c[1:-1] - \
            gamma * c[1:-1] * phi[1:-1] - \
            d_c * c[1:-1]
    else:
        dc_dt[1:-1] = 0

    # dphi_dt
    dphi_dt[1:-1] = D_phi * (phi[:-2] - 2 *\
        phi[1:-1] + phi[2:]) / (dx**2) - \
        1 / r * beta_p * (c[1:-1] + c_t[1:-1]) * phi[1:-1] + \
        beta_m * (phi_c[1:-1] + phi_c_t[1:-1])

    # dphi_c_dt
    dphi_c_dt[1:-1] = D_phi_c * (phi_c[:-2] -\
        2 * phi_c[1:-1] + phi_c[2:]) / (dx**2) - \
        chi * dphi_c_dx[1:-1] * dc_dx[1:-1] - \
        chi * phi_c[1:-1] * (c[:-2] - 2 * c[1:-1] + c[2:]) / (dx**2) + \
        (1 / r) * beta_p * c[1:-1] * phi[1:-1] - \
        beta_m * phi_c[1:-1]

    dc_t_dt[1:-1] = D_c_t * (c_t[:-2] - 2 * c_t[1:-1] + c_t[2:]) / (dx**2) - \
        Pe_dc_t_dx[1:-1] - \
        N_R7 * beta_p * c_t[1:-1] * phi[1:-1] + \
        N_R7 * r * beta_m * phi_c_t[1:-1] + \
        gamma * c[1:-1] * phi[1:-1] - \
        d_c_t * c[1:-1]

    dphi_c_t_dt[1:-1] = D_phi_c_t * (phi_c_t[:-2] -\
        2 * phi_c_t[1:-1] + phi_c_t[2:]) / (dx**2) - \
        chi_t * dphi_c_t_dx[1:-1] * dc_t_dx[1:-1] - \
        chi_t * phi_c_t[1:-1] * (c_t[:-2] - 2 * c_t[1:-1] \
        + c_t[2:]) / (dx**2) + \
        (1 / r) * beta_p * c_t[1:-1] * phi[1:-1] - \
        beta_m * phi_c_t[1:-1]

    # Boundary Conditions
    phi[:1] = phi[1:2]     
    phi[-1:] = phi[-2:-1]       

    phi_c[:1] = phi_c[1:2] / (1 + (chi / D_phi_c) * (c[1:2] - c[:1]))  
    phi_c[-1:] = phi_c[-2:-1] / \
        (1 - (chi / D_phi_c) * (c[-1:] - c[-2:-1])) 
        
    phi_c_t[:1] = phi_c_t[1:2] / (1 + (chi_t / D_phi_c_t) * (c_t[1:2]-c_t[:1]))
    phi_c_t[-1:] = phi_c_t[-2:-1] / \
        (1 - (chi_t / D_phi_c_t) * (c_t[-1:] - c_t[-2:-1])) 

    print(f"t:{t}")

    return np.concatenate([dc_dt, dphi_dt, dphi_c_dt, dc_t_dt, dphi_c_t_dt])


# Solve ODE using solve_ivp
initial_conditions = np.concatenate([c_0, phi_0, phi_c_0, c_t_0, phi_c_t_0])
solution = solve_ivp(
    pde, [t_0, t_1], initial_conditions, t_eval=t, method='RK45')

# Store data as matricies
c_matrix = np.zeros((Dt, Dx))
c_t_matrix = np.zeros((Dt, Dx))
phi_matrix = np.zeros((Dt, Dx))
phi_c_matrix = np.zeros((Dt, Dx))
phi_c_t_matrix = np.zeros((Dt, Dx))
phi_combined_matrix = np.zeros((Dt, Dx))
    
# Store fluxes for each time step and spatial point
flux_c_matrix = np.zeros((Dt, Dx))
flux_phi_matrix = np.zeros((Dt, Dx))
flux_phi_c_matrix = np.zeros((Dt, Dx))
flux_c_t_matrix = np.zeros((Dt, Dx))
flux_phi_c_t_matrix = np.zeros((Dt, Dx))
flux_phi_combined_matrix = np.zeros((Dt, Dx))

#%%
for i, t_i in enumerate(t):
    c = solution.y[:Dx, i]
    phi = solution.y[Dx:2*Dx, i]
    phi_c = solution.y[2*Dx:3*Dx, i]
    c_t = solution.y[3*Dx:4*Dx, i]
    phi_c_t = solution.y[4*Dx:5*Dx, i]

    c_matrix[i, :] = c
    phi_matrix[i, :] = phi
    phi_c_matrix[i, :] = phi_c
    c_t_matrix[i, :] = c_t
    phi_c_t_matrix[i, :] = phi_c_t
    phi_combined_matrix[i, :] = phi + phi_c + phi_c_t

    # Calculate fluxes
    flux_c = -1 * np.gradient(c, dx) + Pe*c
    flux_phi = -D_phi * np.gradient(phi, dx)
    flux_phi_c = -D_phi_c * \
        np.gradient(phi_c, dx) + chi * phi_c * np.gradient(c, dx)
    flux_c_t = -D_c_t * np.gradient(c_t, dx) + Pe*c_t
    flux_phi_c_t = -D_phi_c_t * \
        np.gradient(phi_c_t, dx) + chi_t * phi_c_t * np.gradient(c_t, dx)
    
    flux_c_matrix[i, :] = flux_c 
    flux_phi_matrix[i, :] = flux_phi
    flux_phi_c_matrix[i, :] = flux_phi_c
    flux_c_t_matrix[i, :] = flux_c_t
    flux_phi_c_t_matrix[i, :] = flux_phi_c_t
    flux_phi_combined_matrix[i, :] = flux_phi + flux_phi_c + flux_phi_c_t
    

#%% Store data 
np.savetxt("c_matrix.txt", c_matrix, delimiter=",")
np.savetxt("phi_matrix.txt", phi_matrix, delimiter=",")
np.savetxt("phi_c_matrix.txt", phi_c_matrix, delimiter=",")
np.savetxt("c_t_matrix.txt", c_t_matrix, delimiter=",")
np.savetxt("phi_c_t_matrix.txt", phi_c_t_matrix, delimiter=",")
np.savetxt("phi_combined_matrix.txt", phi_combined_matrix, delimiter=",")

np.savetxt("flux_c_matrix.txt", flux_c_matrix, delimiter=",")
np.savetxt("flux_phi_matrix.txt", flux_phi_matrix, delimiter=",")
np.savetxt("flux_phi_c_matrix.txt", flux_phi_c_matrix, delimiter=",")
np.savetxt("flux_c_t_matrix.txt", flux_c_t_matrix, delimiter=",")
np.savetxt("flux_phi_c_t_matrix.txt", flux_phi_c_t_matrix, delimiter=",")
np.savetxt("flux_phi_combined_matrix.txt", flux_phi_combined_matrix, delimiter=",")

