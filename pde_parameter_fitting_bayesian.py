# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:43:17 2025

@author: 44788
"""

import scipy.stats as stats
import numpy as np
from scipy.integrate import solve_ivp 
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
import seaborn as sns


def CalculateError(chemokine_present, flow_direction, D_phi, chi):
    
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
       Pe = 2
       proportion_bound = 0
       cell_count = 78
    else:
        if flow_direction == 'NEG':
            data_file = "M12_wDC_CCL21_NEG_pos_export.txt"
            density_data_file = "M12_wDC_CCL21_NEG_density_data.txt" 
            flux_data_file = "M12_wDC_CCL21_NEG_flux_data.txt"
            Pe = -2
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
    phi_0 = phi0(x)
    phi_c_0 = phic0(x)
    
    t_0 =   0                                      # t_0 : Initial time                                    # t_1 : Final time                                                                                       
    t_1 = 1800/(1.44e4)                            # dt: Time step.
    dt = t_1/100 
    Dt =    int((t_1 - t_0 + dt) / dt)             # Dt: Number of time steps.
    t =     np.linspace(t_0, t_1, Dt)              # t : Time mesh.
    
    # Define PDE as a System of ODEs
    def pde(t, y):
        
        c = y[:Dx]
        phi = y[Dx:2*Dx]
        phi_c = y[2*Dx:]
    
        dc_dt = np.zeros_like(c)
        dphi_dt = np.zeros_like(phi)
        dphi_c_dt = np.zeros_like(phi_c)
        
        Pe_dc_dx = np.zeros(Dx)
        for i in range(1, Dx-1): 
            
            # Backward difference if Pe > 0
            # Forward difference  if Pe < 0 
            if Pe > 0:
                Pe_dc_dx[i] = Pe * (c[i] - c[i-1]) / dx
            else:
                Pe_dc_dx[i] = Pe * (c[i+1] - c[i]) / dx
                
        dphi_c_dx = np.zeros(Dx)
        dc_dx = np.zeros(Dx)
        for i in range(1, Dx-1):
            dc_dx_test = (c[i+1] - c[i-1]) / (2*dx) 
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
        
        c = solution.y[:Dx, i]  
        phi = solution.y[Dx:2*Dx, i]  
        phi_c = solution.y[2*Dx:, i]  
        
        # Assign to matrices with time (t_i) as rows and space (x) as columns
        phi_matrix[i, :] = phi 
        phi_c_matrix[i, :] = phi_c
    
        # Calculate fluxes
        flux_phi = -D_phi * np.gradient(phi, dx) 
        flux_phi_c = -D_phi_c * np.gradient(phi_c, dx) + chi * phi_c * np.gradient(c, dx) 

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
    
    return phi_combined_matrix, error_density, flux_phi_combined_matrix, error_flux 

def load_data(filename):
    return np.loadtxt(filename, delimiter=",")

# Function to compute the log-likelihood
def log_likelihood(F, y, sigma=100):
    diff = F[:-1, :] - y
    return (-1/2*sigma**2)*np.sum(diff** 2) - (np.prod(y.shape)/2)*np.log(2*np.pi*sigma**2)

def log_prior(D, chi):
    if D <= 0 or chi <= 0:  # Ensure parameters remain positive
        return -np.inf
    
    log_P_D = stats.norm(loc=1, scale=1).logpdf(D)  
    log_P_chi = stats.norm(loc=6, scale=1).logpdf(chi) 

    return log_P_D + log_P_chi


# Symmetric proposal distribution
def propose_theta(theta, step_size):
    return np.random.normal(0, step_size, size=theta.shape)


def metropolis_hastings(data_filename, num_samples, step_size, chemokine_present,
                        flow_direction):
    # Load experimental data
    y = load_data(data_filename)

    # Initial parameter values
    D, chi = 0.01, 0.5
    theta = np.array([D, chi])

    # Run model with initial parameters
    densityMatrix_current, error_density, fluxMatrix_current, error_flux =\
        CalculateError(chemokine_present, flow_direction, theta[0], theta[-1])
    #log_like_current = log_likelihood(densityMatrix_current, y)
    log_like_current = log_likelihood(fluxMatrix_current, y)
    log_prior_current = log_prior(theta[0], theta[1])
    log_posterior_current = log_like_current# + log_prior_current
    
    accepted_samples = [theta]
    for i in range(num_samples):
        print(f"\nIteration: {i}\n")

        # Propose new parameters
        theta_star = propose_theta(theta, step_size)
        D_star, chi_star = theta_star
        print(f"Proposed parameters: {theta_star}")

        # Ensure proposed values are within valid ranges
        if D_star <= 0 or chi_star <= 0:
            print("Rejected due to non-positive parameters.")
            continue  # Skip storing and plotting rejected parameters

        # Run model with proposed parameters
        print("Running simulation for proposed parameters...")
        densityMatrix_star, error_density, fluxMatrix_star, error_flux =\
            CalculateError(chemokine_present, flow_direction, D_star, chi_star)
        
        # Compute log-likelihood and log-prior for the proposed state
        #log_like_star = log_likelihood(densityMatrix_star, y)
        log_like_star = log_likelihood(fluxMatrix_star, y)
        log_prior_star = log_prior(D_star, chi_star)
        log_posterior_star = log_like_star# + log_prior_star

        # Compute acceptance probability 
        log_r = log_posterior_star - log_posterior_current
        print(f"post star {log_posterior_star}")
        print(f"post current {log_posterior_current}")
        print(f"Acceptance ratio (exp(log_r)): {np.exp(log_r)}")

        # Accept or reject the proposal
        random_var = random.uniform(0, 1)
        log_random_var = np.log(random_var)
        print(f"random variable for comparison: {random_var}")

        if log_random_var < log_r:
            theta = theta_star  
            log_posterior_current = log_posterior_star
            log_like_current = log_like_star
            densityMatrix_current = densityMatrix_star
            fluxMatrix_current = fluxMatrix_star
            print("Proposal accepted.")
        else:
            print("Proposal rejected.")
            
        accepted_samples.append(theta)  # Store only accepted values
        
        np.savetxt("parameter_list.txt", accepted_samples, delimiter=",")
        
        # Plot only accepted samples
        if accepted_samples:
            accepted_samples_np = np.array(accepted_samples)
            plt.figure(figsize=(8, 6))

            # Plot D
            plt.subplot(2, 1, 1)
            plt.ylabel(r'$D$')
            plt.plot(range(len(accepted_samples_np)), accepted_samples_np[:, 0],
                     linestyle='-', color='darkred')
            plt.grid()
            
            # Plot chi
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accepted_samples_np)), accepted_samples_np[:, 1],
                     linestyle='-', color='darkred')
            plt.xlabel('Iteration')
            plt.ylabel(r'$\chi$')
            plt.grid()

            plt.tight_layout()
            plt.show()

    return np.array(accepted_samples)

# Example function call
parameters = metropolis_hastings("M12_wDC_CCL21_POS_flux_data.txt", 10000, 
                                 (0.1, 1), True, 'POS')

#parameters = np.loadtxt("parameter_list.txt", delimiter=",")

plt.figure(figsize=(6, 4))

# Plot D timeseries
plt.subplot(2, 1, 1)
plt.plot(range(len(parameters)), parameters[:, 0], linestyle='-', 
         color='darkred')
plt.ylabel(r'$D$')
plt.ylim(0,0.4)
plt.grid()

# Plot chi timeseries
plt.subplot(2, 1, 2)
plt.plot(range(len(parameters)), parameters[:, 1], linestyle='-',
         color='darkred')
plt.xlabel('Iteration')
plt.ylabel(r'$\chi$')
plt.ylim(0,2)
plt.grid()

plt.tight_layout()
plt.show()
plt.close()

# Remove the first 10% of values (burn-in period)
parameters = parameters[int(len(parameters)/10):]


num_prior_samples = 5000  
D_prior_samples = np.random.normal(loc=1, scale=1, size=num_prior_samples)
chi_prior_samples = np.random.normal(loc=6, scale=1, size=num_prior_samples)

D_prior_samples = np.array(D_prior_samples)
chi_prior_samples = np.array(chi_prior_samples)

kde_D = gaussian_kde(parameters[:, 0], bw_method='scott')
D_values = np.linspace(0, 30, 10000)
D_density = kde_D(D_values)
D_modal_value = D_values[np.argmax(D_density)]  

# KDE for chi (parameters[:, 1])
kde_chi = gaussian_kde(parameters[:, 1], bw_method='scott')
chi_values = np.linspace(0, 30, 10000)
chi_density = kde_chi(chi_values)
chi_modal_value = chi_values[np.argmax(chi_density)]

print(D_modal_value)
print(chi_modal_value)

# Plot KDEs
plt.figure(figsize=(6, 3.5))

# KDE for D
plt.subplot(2, 1, 1)
Dplot = sns.kdeplot(parameters[:, 0], color='darkred', fill=True,
                    label='Posterior')
plt.xlabel(r'$D$')
plt.xlim(0,0.3)
plt.grid()

# KDE for chi
plt.subplot(2, 1, 2)
chiplot = sns.kdeplot(x=parameters[:, 1], color='darkred', fill=True,
                      label='Posterior')
plt.xlabel(r'$\chi$')
plt.xlim(0,2)
plt.grid()

plt.tight_layout()
plt.show()


