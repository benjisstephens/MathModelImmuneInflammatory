# Plot the three-variable PDE results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
 
steady_state = False

t_0 =   0              
if steady_state:
    t_1 = 30
else:                                 # t_0 : Initial time
    t_1 =   1800/(1.44e4)             # t_1 : Final time
dt = t_1/100
Dt =    int((t_1 - t_0 + dt) / dt)    # Dt: Number of time steps.
t =     np.linspace(t_0, t_1, Dt)     # t : Time mesh.

# Define spatial step and domain.
x_0 =   0                             # x_0 : Left boundary x
x_1 =   1                             # x_1 : Right boundary x
dx =    0.01                          # dx: Spatial step.
Dx =    int((x_1 - x_0 + dx) / dx)    # Dx: Number of spatial steps.
x =     np.linspace(x_0, x_1, Dx)     # x : Spatial mesh.

toggle_save = True

plt.rcParams["font.size"] = 14

#%% Read data
c_matrix = np.loadtxt("c_matrix.txt", delimiter=",")
phi_matrix = np.loadtxt("phi_matrix.txt", delimiter=",")
phi_c_matrix = np.loadtxt("phi_c_matrix.txt", delimiter=",")
phi_combined_matrix = np.loadtxt("phi_combined_matrix.txt", delimiter=",")
flux_c_matrix = np.loadtxt("flux_c_matrix.txt", delimiter=",")
flux_phi_matrix = np.loadtxt("flux_phi_matrix.txt", delimiter=",")
flux_phi_c_matrix = np.loadtxt("flux_phi_c_matrix.txt", delimiter=",")
flux_phi_combined_matrix = np.loadtxt("flux_phi_combined_matrix.txt",
                                      delimiter=",")

#%%# Plot just results

for i, t_i in enumerate(t):
    if i % 100 == 0:  # Process every 300th time step for better visualization
        # Create figure for concentrations
        fig, ax = plt.subplots(figsize=(4,4), dpi=300)

        # Plot concentrations
        ax.plot(x, c_matrix[i, :], label='c', color='darkmagenta')
        ax.plot(x, phi_matrix[i, :]+phi_c_matrix[i, :],
                label=r'$\phi$+$\phi_c$', color='black', linestyle='-')
        ax.plot(x, phi_matrix[i, :], label=r'$\phi$', color='darkgreen')
        ax.plot(x, phi_c_matrix[i, :], label=r'$\phi_c$',
                color='darkgreen', linestyle='--')

        # Set axis labels, limits, and grid
        ax.set_xlabel('x')
        ax.set_xlim(x_0, x_1)
        ax.set_xticks(np.linspace(x_0, x_1, 5))
        ax.set_ylim(0, 1)
        ax.grid()

        # Adjust layout
        plt.tight_layout()

        # Optionally save the figure
        if toggle_save:
            plt.savefig(f"c_phi_phi_c_{t_i:.3f}s.png",
                        dpi=300, bbox_inches='tight')
        
        plt.show()

    
            
#%%# Plot results for density and flux
for i, t_i in enumerate(t):
    if i % 100 == 0:  # Process every 100th time step for better visualization 
        # Create figure and subplots for concentrations and fluxes
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
        
        # Plot concentrations
        axs[0].plot(x, c_matrix[i, :], label='c', color='darkmagenta')
        axs[0].plot(x, phi_matrix[i, :], label=r'$\phi$', color='darkgreen')
        axs[0].plot(x, phi_c_matrix[i, :], label=r'$\phi_c$', color='darkgreen',
                    linestyle='--')
        axs[0].plot(x, phi_combined_matrix[i, :], label=r'$\phi$+$\phi_c$', 
                    color='black', linestyle='-')
        axs[0].set_xlabel('x')
        axs[0].set_xlim(x_0, x_1)
        axs[0].set_xticks(np.linspace(x_0, x_1, 5))
        axs[0].set_ylim(0, 1)
        axs[0].grid()
        axs[0].set_title("Densities",fontsize=14)
        
        # Plot fluxes
        axs[1].plot(x, flux_c_matrix[i, :], color='darkmagenta')
        axs[1].plot(x, flux_phi_matrix[i, :], color='darkgreen')
        axs[1].plot(x, flux_phi_c_matrix[i, :], color='darkgreen', linestyle='--')
        #axs[1].plot(x, flux_phi_combined_matrix[i, :]+flux_phi_c_matrix[i, :], 
        #   label=r'$\phi$+$\phi_c$', color='black', linestyle='-')
        axs[1].set_xlabel('x')
        axs[1].set_xlim(x_0, x_1)
        axs[1].set_xticks(np.linspace(x_0, x_1, 5))
        axs[1].set_ylim(-0.035, 0.035)
        axs[1].axhline(y=0, color='black')
        axs[1].grid()
        axs[1].set_title("Fluxes",fontsize=14)
        
        # Combine legends into one, positioned above the plots
        handles, labels = axs[0].get_legend_handles_labels()
        handles_flux, labels_flux = axs[1].get_legend_handles_labels()
        handles_combined = handles + handles_flux
        labels_combined = labels + labels_flux
        #fig.legend(handles_combined, labels_combined, 
        # loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
        
        plt.tight_layout()
        plt.savefig(f"c_phi_phi_c_flux_{t_i:.3f}s.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        
#%% Plot density matrices for phi, phi_c, and c in one loop

density_matrices = [c_matrix*6, phi_matrix*0.915,
                    phi_c_matrix*0.915, phi_combined_matrix*0.915]
labels = ['Density c', 'Density phi', 'Density phi_c', 'Density phi_combined']
file_names = ['c_matrix', 'phi_matrix', 'phi_c_matrix', 'phi_combined_matrix']

for density_matrix, label, file_name in zip(density_matrices,
                                            labels, file_names):
    if file_name == 'c_matrix':
        vmin, vmax = 0, 6
    else:
        vmin, vmax = 0, 0.4
    
    fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7),
                            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)

    im = ax.imshow(
        density_matrix,
        extent=[x_0, x_1, t_0, t_1],
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=vmin,
        vmax=vmax
    )
    
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
        ax.set_yticklabels(np.linspace(1.44e4*t_0,
                                       1.44e4*t_1, 7).astype(int))
    ax.set_xticklabels(np.linspace(0, 1200, 7).astype(int))
    
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                        location='top')
    cbar.ax.tick_params(labelsize=14)
    if file_name == 'c_matrix':
        cbar.set_label('Density ($CCL21/\mu m$)', fontsize=14)
    else:
        cbar.set_label('Density ($cells/\mu m$)', fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0)) 
    cbar.ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout(pad=1.0)
    plt.savefig(f"{file_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

        
#%% Plot flux matrices for phi, phi_c, and c in one loop

flux_matrices = [flux_c_matrix*8e4, flux_phi_matrix*7.9e-4, 
                 flux_phi_c_matrix*7.9e-4, flux_phi_combined_matrix*7.9e-4]
labels = ['Flux of c', 'Flux of phi', 'Flux of phi_c', 'Flux of phi combined']
file_names = ['flux_c_matrix', 'flux_phi_matrix', 'flux_phi_c_matrix',
              'flux_phi_combined_matrix']

for flux_matrix, label, file_name in zip(flux_matrices, labels, file_names):
    
    if file_name == 'flux_c_matrix':
        vlim = 1e5
    else:
        vlim = 1e-4
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
        ax.set_yticklabels(np.linspace(1.44e4 / 3600 * \
                                     t_0, 1.44e4 / 3600 * t_1, 7).astype(int))
    else:
        ax.set_yticklabels(np.linspace(1.44e4 *\
                                       t_0, 1.44e4 * t_1, 7).astype(int))
    
    # Create and configure the colorbar
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                        location='top')
    cbar.ax.tick_params(labelsize=14)
    if file_name == 'flux_c_matrix':
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
