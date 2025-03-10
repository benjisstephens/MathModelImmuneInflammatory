# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:09:57 2025

@author: 44788
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

chemokine_present = False
flow_direction = 'NEG'
steady_state = False

# Define spatial step and domain.
x_0 = 0                              # x_0 : Left boundary x
x_1 = 1                              # x_1 : Right boundary x
dx = 0.01                            # dx: Spatial step.
Dx = int((x_1 - x_0 + dx) / dx)      # Dx: Number of spatial steps.
x = np.linspace(x_0, x_1, Dx)        # x : Spatial mesh.

t_0 =   0                                           # t_0 : Initial time
if steady_state == True:                               
    t_1 = 30                                        # t_1 : Final time                                    
else:                                                       
    t_1 = 1800/(1.44e4)                            # dt: Time step.
dt = t_1/100 
Dt =    int((t_1 - t_0 + dt) / dt)                  # Dt: Number of time steps.
t =     np.linspace(t_0, t_1, Dt)                   # t : Time mesh.

plt.rcParams["font.size"] = 14

#%% Read data
c_matrix = np.loadtxt("c_matrix.txt", delimiter=",")
phi_matrix = np.loadtxt("phi_matrix.txt", delimiter=",")
phi_c_matrix = np.loadtxt("phi_c_matrix.txt", delimiter=",")
c_t_matrix = np.loadtxt("c_t_matrix.txt", delimiter=",")
phi_c_t_matrix = np.loadtxt("phi_c_t_matrix.txt", delimiter=",")
phi_combined_matrix = np.loadtxt("phi_combined_matrix.txt", delimiter=",")

flux_c_matrix = np.loadtxt("flux_c_matrix.txt", delimiter=",")
flux_phi_matrix = np.loadtxt("flux_phi_matrix.txt", delimiter=",")
flux_phi_c_matrix = np.loadtxt("flux_phi_c_matrix.txt", delimiter=",")
flux_c_t_matrix = np.loadtxt("flux_c_t_matrix.txt", delimiter=",")
flux_phi_c_t_matrix = np.loadtxt("flux_phi_c_t_matrix.txt", delimiter=",")
flux_phi_combined_matrix = np.loadtxt("flux_phi_combined_matrix.txt",
                                      delimiter=",")

#%% Plot just results
for i, t_i in enumerate(t):
    if i % 10 == 0:
        # Plot concentrations
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        
        # Plot concentrations
        ax.plot(x, c_matrix[i, :], label='c', color='darkmagenta')
        ax.plot(x, c_t_matrix[i, :], label=r'$c_t$', color='darkmagenta', 
                linestyle='--')
        ax.plot(x, phi_matrix[i, :], label=r'$\phi$', color='darkgreen')
        ax.plot(x, phi_c_matrix[i, :], label=r'$\phi_c$', color='darkgreen',
                linestyle='--')
        ax.plot(x, phi_c_t_matrix[i, :], label=r'$\phi_c$', color='darkgreen', 
                linestyle='-.')

        # Set axis labels, limits, and grid
        ax.set_xlabel('x')
        ax.set_xlim(x_0, x_1)
        ax.set_xticks(np.linspace(x_0, x_1, 5))
        ax.set_ylim(0, 1)
        ax.grid()

        # Add legend
        #ax.legend(loc='upper right')

        # Adjust layout
        plt.tight_layout()

        # Optionally save the figure
        if toggle_save:
            plt.savefig(f"5variable_density_t{t_i}.png", dpi=300, 
                        bbox_inches='tight')

        plt.show()


# %%# Plot results for density and flux

for i, t_i in enumerate(t):
    if i % 10 == 0:
        # Create figure and subplots for concentrations and fluxes
        fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), dpi=300)

        # Plot concentrations
        axs[0].plot(x, c_matrix[i, :], label='c', color='darkmagenta')
        axs[0].plot(x, c_t_matrix[i, :], label=r'$c_t$',
                    color='mediumvioletred')
        axs[0].plot(x, phi_matrix[i, :], label=r'$\phi$', color='darkgreen')
        axs[0].plot(x, phi_c_matrix[i, :], label=r'$\phi_c$',
                    color='darkgreen', linestyle='--')
        axs[0].plot(x, phi_c_t_matrix[i, :],
                    label=r'$\phi_{c_t}$', color='darkgreen', linestyle='-.')
        axs[0].set_xlabel('x')
        axs[0].set_xlim(x_0, x_1)
        axs[0].set_ylim(0, 1)
        axs[0].grid()
        axs[0].set_title("Densities",fontsize=14)

        # Plot fluxes
        # axs[1].plot(x, flux_c_matrix[i, :], color='darkmagenta')
        axs[1].plot(x, flux_phi_matrix[i, :], color='darkgreen')
        axs[1].plot(x, flux_phi_c_matrix[i, :],
                    color='darkgreen', linestyle='--')
        # axs[1].plot(x, flux_c_t_matrix[i, :], color='mediumvioletred')
        axs[1].plot(x, flux_phi_c_t_matrix[i, :],
                     color='darkgreen', linestyle='-.')
        axs[1].set_xlabel('x')
        axs[1].set_xlim(x_0, x_1)
        axs[1].set_ylim(-0.1, 0.1)
        axs[1].axhline(y=0, color='black')
        axs[1].grid()
        axs[1].set_title("Fluxes",fontsize=14)

        # Combine legends into one, positioned above the plots
        handles, labels = axs[0].get_legend_handles_labels()
        handles_flux, labels_flux = axs[1].get_legend_handles_labels()
        handles_combined = handles + handles_flux
        labels_combined = labels + labels_flux
        #fig.legend(handles_combined, labels_combined,
                   #loc='upper center', ncol=3, frameon=False)

        # Adjust spacing to accommodate the legend
        plt.tight_layout()

        # Optionally save the figure
        if toggle_save:
            plt.savefig(f"5variable_t{t_i:.1f}s.png", dpi=300, 
                        bbox_inches='tight')

        plt.show()




# %% Plot density matrices for phi, phi_c, and c in one loop

density_matrices = [c_matrix*6, phi_matrix*0.915, phi_c_matrix*0.915,
                    c_t_matrix*6, phi_c_t_matrix*0.915, 
                    phi_combined_matrix*0.915]
labels = ['Density c', 'Density phi', 'Density phi_c', 'Density c_t',
          'Density phi_c_t','Density phi_combined']
file_names = ['c_matrix', 'phi_matrix', 'phi_c_matrix', 'c_t_matrix', 
              'phi_c_t_matrix','phi_combined_matrix']

for density_matrix, label, file_name in zip(density_matrices, 
                                            labels, file_names):    
    if file_name == 'c_matrix':
        vmin, vmax = 0, 6 
    elif file_name == 'phi_matrix':
        vmin, vmax = 0, 0.6
    elif file_name == 'phi_c_matrix':
        vmin, vmax = 0, 0.6
    elif file_name == 'c_t_matrix':
        vmin, vmax = 0, 1#6e-2
    elif file_name == 'phi_c_t_matrix':
        vmin, vmax = 0, 6e-3
    else:
        vmin, vmax = 0, 0.4
    
    fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7),
                        gridspec_kw={"height_ratios": [1, 20]}, dpi=300)

    # Create the image plot
    im = ax.imshow(
        density_matrix,
        extent=[x_0, x_1, t_0, t_1],
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=vmin,
        vmax=vmax
    )
    
    # Set axis labels
    ax.set_xlabel('Position $(\mu m)$')
    if steady_state:
        ax.set_ylabel('Time (hrs)')
    else:
        ax.set_ylabel('Time (s)')
    
    ax.set_xticks(np.linspace(x_0, x_1, 7))
    ax.set_yticks(np.linspace(t_0, t_1, 7))
    if steady_state:
        ax.set_yticklabels(np.linspace(1.44e4/3600*t_0,
                                       1.44e4/3600*t_1, 7).astype(int))
    else:
        ax.set_yticklabels(np.linspace(1.44e4*t_0, 1.44e4*t_1, 7).astype(int))
    ax.set_xticklabels(np.linspace(0, 1200, 7).astype(int))
    
    # Create and configure the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        location='top')
    cbar.ax.tick_params(labelsize=14)
    if file_name == 'c_matrix' or file_name == 'c_t_matrix':
        cbar.set_label('Density $(CCL21/\mu m)$', fontsize=14)
    else:
        cbar.set_label('Density $(cells/\mu m)$', fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Force scientific notation (e.g., 1e0)
    cbar.ax.xaxis.set_major_formatter(formatter)
    
    # Adjust layout
    plt.tight_layout(pad=1.0)
    plt.savefig(f"{file_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    

# %% Plot flux matrices for phi, phi_c, and c in one loop

flux_matrices = [flux_c_matrix*8e4, flux_phi_matrix*7.9e-4, 
                 flux_phi_c_matrix*7.9e-4, flux_c_t_matrix*8e5, 
                 flux_phi_c_t_matrix*7.9e-4, flux_phi_combined_matrix*7.9e-4]
labels = ['Flux of c', 'Flux of phi', 'Flux of phi_c', 'Flux of c_t',
          'Flux of phi_c_t', 'Flux of phi_combined']
file_names = ['flux_c_matrix', 'flux_phi_matrix', 'flux_phi_c_matrix', 
              'flux_c_t_matrix','flux_phi_c_t_matrix', 
              'flux_phi_combined_matrix']

for flux_matrix, label, file_name in zip(flux_matrices, labels, file_names):
    
    if file_name == 'flux_c_matrix':
        vlim = 1e6
    elif file_name == 'flux_phi_matrix':
        vlim = 1e-4
    elif file_name == 'flux_phi_c_matrix':
        vlim = 1e-4
    elif file_name == 'flux_c_t_matrix':
        vlim = 1e6
    elif file_name == 'flux_phi_c_t_matrix':
        vlim = 1e-6
    else:
        vlim = 2e-5

    vmin, vmax = -vlim, vlim
    
    fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7), 
                            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)

    # Create the image plot
    im = ax.imshow(
        flux_matrix,
        extent=[x_0, x_1, t_0, t_1],
        aspect='auto',
        origin='lower',
        cmap='seismic',
        vmin=vmin,
        vmax=vmax
    )
    
    # Set axis labels
    ax.set_xlabel('Position $(\mu m)$')
    if steady_state:
        ax.set_ylabel('Time (hrs)')
    else:
        ax.set_ylabel('Time (s)')
    
    ax.set_xticks(np.linspace(x_0, x_1, 7))
    ax.set_yticks(np.linspace(t_0, t_1, 7))
    ax.set_xticklabels(np.linspace(1200 * x_0, 1200 * x_1, 7).astype(int))
    if steady_state:
        ax.set_yticklabels(np.linspace(1.44e4 / 3600 *\
                                      t_0, 1.44e4 / 3600 * t_1, 7).astype(int))
    else:
        ax.set_yticklabels(np.linspace(1.44e4 *\
                                       t_0, 1.44e4 * t_1, 7).astype(int))
    
    # Create and configure the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        location='top')
    cbar.ax.tick_params(labelsize=14)
    if file_name == 'flux_c_matrix' or file_name == 'flux_c_t_matrix':
        cbar.set_label('Flux ($CCL21/s$)', fontsize=14)
    else:
        cbar.set_label('Flux ($cells/s$)', fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 3))
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  
    cbar.ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout(pad=1.0)
    plt.savefig(f"{file_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    