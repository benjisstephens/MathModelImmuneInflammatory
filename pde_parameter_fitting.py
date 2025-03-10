# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:43:17 2025

@author: 44788
"""

import numpy as np
from scipy.integrate import solve_ivp 
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def CalculateError(chemokine_present, flow_direction, D_phi, chi):
    
    # Define spatial step and domain.
    x_0 =   0                             # x_0 : Left boundary x
    x_1 =   1                             # x_1 : Right boundary x
    dx =    0.01                          # dx: Spatial step.
    Dx =    int((x_1 - x_0 + dx) / dx)    # Dx: Number of spatial steps.
    x =     np.linspace(x_0, x_1, Dx)     # x : Spatial mesh.
    
    D_phi_c =   D_phi                
    N_R7 =      30000                   
    r =         9.8e-7                  
    d_c =       1.9e-2                    
    beta_p =    1.5e-4                    
    beta_m =    72                        
                     
    # File to be processed
    if chemokine_present == False:
       data_file = "M4_wDC_CTRL_POS_pos_export.txt" 
       density_data_file = "M4_wDC_CTRL_POS_density_data.txt" 
       flux_data_file = "M4_wDC_CTRL_POS_flux_data.txt" 
       Pe = -2
       proportion_bound = 0
       cell_count = 78
    else:
        if flow_direction == 'NEG':
            data_file = "M12_wDC_CCL21_NEG_pos_export.txt"
            density_data_file = "M12_wDC_CCL21_NEG_density_data.txt" 
            flux_data_file = "M12_wDC_CCL21_NEG_flux_data.txt"
            Pe = 2
            proportion_bound = 0.2
            cell_count = 573
        elif flow_direction == 'DIF':
            data_file = "M12_wDC_CCL21_DIF_pos_export.txt"
            density_data_file = "M12_wDC_CCL21_DIF_density_data.txt" 
            flux_data_file = "M12_wDC_CCL21_DIF_flux_data.txt"
            Pe = 0
            proportion_bound = 0.36
            cell_count = 143
        elif flow_direction == 'POS':
            data_file = "M12_wDC_CCL21_POS_pos_export.txt"
            density_data_file = "M12_wDC_CCL21_POS_density_data.txt" 
            flux_data_file = "M12_wDC_CCL21_POS_flux_data.txt"
            Pe = -2
            proportion_bound = 0.36
            cell_count = 250
    
    data = pd.read_csv(data_file)
        
    # Determine the first (minimum) time step in the dataset.
    first_time = data['Time(s)'].min()
    
    # Count the number of cells that first appear at the first time step.
    # (Only these cells are used to form the KDE.)
    first_time_cells = data.groupby('CellID')['Time(s)'].min().eq(first_time).sum()
    
    # Group the data by CellID for processing.
    groups = data.groupby('CellID')
    
    # Use the file name (without the suffix) for titling/saving.
    title = data_file.replace('_pos_export.txt', '')
    
    # Loop through each cell and extract its position at the first time step.
    # Only include cells that have their first recorded time equal to first_time.
    new_x_positions = []  # This will be our transformed positions (1250 - y)
    for cell_id, group in groups:
        time_group = group['Time(s)']
        # Only include the cell if its first time equals the overall first_time.
        if first_time < time_group.min() or first_time > time_group.max():
            continue
        
        # Interpolate the y-position at first_time.
        y_group = group['y(microns)']
        y_pos = np.interp(first_time, time_group, y_group)
        
        # Transform the y-position (as in your original code).
        new_x = 1250 - y_pos
        new_x_positions.append(new_x)
        
    new_x_positions = np.array(new_x_positions)
    
    # Set the bandwidth scaling using the number of cells (optimal bandwidth ~ n^(-1/5)).
    kde_1d = gaussian_kde(new_x_positions, bw_method=1 * cell_count**(-1/5))
    
    # Evaluate the KDE on the original domain [0, 1200].
    grid_1d = np.linspace(0, 1200, 200)
    # Multiply by first_time_cells to scale the density to "cells per micron".
    density_1d = kde_1d(grid_1d) * first_time_cells
    
    # Transform the density: if x = 1200*u then f(x)dx = g(u)du with dx/du = 1200.
    density_rescaled = kde_1d(1200 * x) * first_time_cells
    density_interp = interp1d(x, density_rescaled, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
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
    phi_0 = phi0(x)
    phi_c_0 = phic0(x)
    
    t_0 =   0                                           # t_0 : Initial time                                    # t_1 : Final time                                                                                       
    t_1 = 1800/(1.44e4)                            # dt: Time step.
    dt = t_1/100 
    Dt =    int((t_1 - t_0 + dt) / dt)                  # Dt: Number of time steps.
    t =     np.linspace(t_0, t_1, Dt)                   # t : Time mesh.
    
    # Define PDE as a System of ODEs
    def pde(t, y):
        c = y[:Dx]
        phi = y[Dx:2*Dx]
        phi_c = y[2*Dx:]
    
        dc_dt = np.zeros_like(c)
        dphi_dt = np.zeros_like(phi)
        dphi_c_dt = np.zeros_like(phi_c)
        

    
        # Calculate sign of Pe at each step to determine how to do upwind scheme correctly
        
        Pe_dc_dx = np.zeros(Dx)
        for i in range(1, Dx-1): 
            
            # Backward difference if Pe > 0
            # Forward difference  if Pe < 0 
            if Pe > 0:
                Pe_dc_dx[i] = Pe * (c[i] - c[i-1]) / dx
            else:
                Pe_dc_dx[i] = Pe * (c[i+1] - c[i]) / dx
        
                
        # Calculate sign of chi * dc_dx at each step to determine how to do upwind scheme correctly
        dphi_c_dx = np.zeros(Dx)
        dc_dx = np.zeros(Dx)
        for i in range(1, Dx-1):
            dc_dx_test = (c[i+1] - c[i-1]) / (2*dx) # Use central difference to approximate dc_dx sign
            # Backward difference if chi * dc_dx > 0
            # Forward difference  if chi * dc_dx < 0
            if chi*dc_dx_test > 0 :
                dphi_c_dx[i] = (phi_c[i] - phi_c[i-1]) / dx
                dc_dx[i]     = (c[i] - c[i-1]) / dx
            else:
                dphi_c_dx[i] = (phi_c[i+1] - phi_c[i]) / dx
                dc_dx[i]     = (c[i+1] - c[i]) / dx
                
        
        # dc_dt
        if chemokine_present == True:
            dc_dt[1:-1] = (c[:-2] - 2 * c[1:-1] + c[2:]) / (dx**2) - \
                          Pe_dc_dx[1:-1] - \
                          N_R7 * beta_p * c[1:-1] * phi[1:-1] + \
                          N_R7 * r * beta_m * phi_c[1:-1] - \
                          d_c * c[1:-1]
        else:
            dc_dt[1:-1] = 0
            
    
        # dphi_dt
        dphi_dt[1:-1] = D_phi * (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / (dx**2) - \
                          1 / r * beta_p * c[1:-1] * phi[1:-1] + \
                          beta_m * phi_c[1:-1]
    
        # dphi_c_dt
        dphi_c_dt[1:-1] = D_phi_c * (phi_c[:-2] - 2 * phi_c[1:-1] + phi_c[2:]) / (dx**2) - \
                          chi * dphi_c_dx[1:-1] * dc_dx[1:-1] - \
                          chi * phi_c[1:-1] * (c[:-2] - 2 * c[1:-1] + c[2:]) / (dx**2) + \
                          (1 / r) * beta_p * c[1:-1] * phi[1:-1] - \
                          beta_m * phi_c[1:-1]
    
        # Boundary Conditions
        phi[:1] = phi[1:2]          # Left
        phi[-1:] = phi[-2:-1]       # Right
        
        #TWO STEP
        phi_c[:1]  = phi_c[1:2]   / (1 + (chi / D_phi_c) * (c[1:2] - c[:1]))   # Left
        phi_c[-1:] = phi_c[-2:-1] / (1 - (chi / D_phi_c) * (c[-1:] - c[-2:-1]))   # Right
        
        #print(f"t:{t} ")
    
        return np.concatenate([dc_dt, dphi_dt, dphi_c_dt])
    
    # Solve ODE using solve_ivp
    initial_conditions = np.concatenate([c_0, phi_0, phi_c_0])
    solution = solve_ivp(pde, [t_0, t_1], initial_conditions, t_eval=t, method='RK45')
    
    # Store data as matricies
    phi_matrix = np.zeros((Dt, Dx))
    phi_c_matrix = np.zeros((Dt, Dx))
    phi_combined_matrix = np.zeros((Dt, Dx))
    
    # Store fluxes for each time step and spatial point
    flux_phi_matrix = np.zeros((Dt, Dx))
    flux_phi_c_matrix = np.zeros((Dt, Dx))
    flux_phi_combined_matrix = np.zeros((Dt, Dx))
    
    for i, t_i in enumerate(t):
        c = solution.y[:Dx, i]  # Extract concentration (c) at time t_i
        phi = solution.y[Dx:2*Dx, i]  # Extract phi at time t_i
        phi_c = solution.y[2*Dx:, i]  # Extract phi_c at time t_i
        
        # Assign to matrices with time (t_i) as rows and space (x) as columns
        phi_matrix[i, :] = phi 
        phi_c_matrix[i, :] = phi_c
    
        # Calculate fluxes
        flux_phi = -D_phi * np.gradient(phi, dx) 
        flux_phi_c = -D_phi_c * np.gradient(phi_c, dx) + chi * phi_c * np.gradient(c, dx) # Example flux for phi_c

        
        flux_phi_matrix[i, :] = flux_phi
        flux_phi_c_matrix[i, :] = flux_phi_c
        
        phi_combined_matrix[i, :] = (phi+phi_c)*0.951
        flux_phi_combined_matrix[i, :] = (flux_phi + flux_phi_c)*7.9e-4
    
    np.savetxt("flux_phi_combined_matrix.txt", flux_phi_combined_matrix, delimiter=",")
    np.savetxt("phi_combined_matrix.txt", phi_combined_matrix, delimiter=",")
    
    density_pde = pd.read_csv("phi_combined_matrix.txt")
    density_data = pd.read_csv(density_data_file)
    density_pde = np.array(density_pde[:-1])
    density_data = np.array(density_data)
    
    
    flux_pde = pd.read_csv("flux_phi_combined_matrix.txt")
    flux_data = pd.read_csv(flux_data_file)
    flux_pde = np.array(flux_pde[:-1])
    flux_data = np.array(flux_data)


    error_density = np.sum((density_pde - density_data) ** 2)
    error_flux = np.sum((flux_pde - flux_data) ** 2)
    
    return density_data, error_density, flux_data, error_flux 

def optimise_parameters(chemokine_present, flow_direction, method, match):
    epsilon_D = 1e-3
    epsilon_chi = 1e-4
    if match == 'density':
        learning_rate_D = 1e-5
        learning_rate_chi = 1e-3
    else:
        learning_rate_D = 1e1
        learning_rate_chi = 1e2
    lambda_LM = 1e5#Initial damping factor for LM
    lambda_factor = 2# LM adjustment factor
    tol = 1e-12  # Prevent negative values
    
    linestyle = '-' if method == "GD" else '-'
    linecolor = "darkred" if match == "density" else "darkblue"
    
    D = 0.01
    chi = 4e-3
    
    D_array = []
    chi_array = []
    error_array = []
    
    if match == 'density':
        iterations = 50
    else:
        iterations = 30
    
    for i in range(iterations):
        
        _, error_d, _, error_f = CalculateError(chemokine_present, flow_direction,  D, chi)
        error = error_d if match == "density" else error_f
        
        D_array.append(D)
        chi_array.append(chi)
        error_array.append(error)
        
        print(f"Iteration: {i}, D: {D}, chi: {chi}, error: {error}")
        
        _, Derror_d, _, Derror_f = CalculateError(chemokine_present, flow_direction, D + epsilon_D, chi)
        _, Cerror_d, _, Cerror_f = CalculateError(chemokine_present, flow_direction, D, chi + epsilon_chi)
        
        Derror = Derror_d if match == "density" else Derror_f
        Cerror = Cerror_d if match == "density" else Cerror_f
        
        grad_D = (Derror - error) / epsilon_D
        grad_chi = (Cerror - error) / epsilon_chi
        
        if method == "GD":
            D -= learning_rate_D * grad_D
            chi -= learning_rate_chi * grad_chi

        else:  # Levenberg-Marquardt
            print(f"lambda: {lambda_LM}")
            H_D = grad_D ** 2
            H_chi = grad_chi ** 2
            
            delta_D = -grad_D / (H_D + lambda_LM)
            delta_chi = -grad_chi / (H_chi + lambda_LM)
            
            D_new = max(D + delta_D, tol)
            chi_new = max(chi + delta_chi, tol)
            
            _, new_error_d, _, new_error_f = CalculateError(chemokine_present, flow_direction,  D_new, chi_new)
            new_error = new_error_d if match == "density" else new_error_f
            
            if new_error < error:
                D, chi = D_new, chi_new
                lambda_LM /= lambda_factor
                error = new_error 
            else:
                lambda_LM *= lambda_factor
        
        D = max(D, tol)
        chi = max(chi, tol)
        
        if D == tol and chi == tol:
            D_array.append(D)
            chi_array.append(chi)
            error_array.append(error)  
            break
        
        plt.figure()
        plt.plot(D_array, chi_array, marker='o')
        
        plt.xlabel("D")
        plt.ylabel("$\chi$")  
        plt.grid(True)
        plt.show()
    
    
    data = np.stack((D_array, chi_array, error_array))
    
    
    
    np.savetxt("D_chi_data.txt", data, delimiter=",", header="D,chi,error", comments='')

    return data, linecolor, linestyle

for method in ["LM","GD"]:
    for chemokine_present, flow_direction in [[True, 'POS'],[True, 'NEG'],[False, 'POS']]:
        
        dataset_density, linecolor_density, linestyle_density = optimise_parameters(chemokine_present, flow_direction, method, match="density")
        dataset_flux, linecolor_flux, linestyle_flux = optimise_parameters(chemokine_present, flow_direction, method, match="flux")
      
        plt.figure(figsize=(6,4),dpi=200)
        plt.plot(dataset_flux[0], dataset_flux[1], marker='o', linestyle=linestyle_flux, markersize=3, color=linecolor_flux)
        plt.plot(dataset_density[0], dataset_density[1], marker='o', linestyle=linestyle_flux, markersize=3, color=linecolor_density)
        
        plt.xlabel("D")
        plt.ylabel("$\chi$")  
        plt.grid(True)
        plt.savefig("D_chi_.png")
        plt.show()
        
    

