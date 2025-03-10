# Plot 2D KDEs and 1D KDEs for each dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import gaussian_kde

plt.rcParams.update({'font.size': 14})  
plt.minorticks_on()

# Read the file names from 'all_data_set_names.txt'
with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

plot_option = 2  # 0: density only, 1: scatter only, 2: both
toggle_save = True
bandwidth_scaling = 1  

div_cell_count = False #Divide the 2D KDE by the number of cells at each time
rescaled = False #Rescale the 1D KDE so that the number of cells is constant

density1d_vmin, density1d_vmax = 0, 1

cmap = get_cmap('Spectral')

# Set density scale limits
if div_cell_count:
    density_vmin, density_vmax = 0, 2e-6
    colorbar_ticks = np.linspace(density_vmin, density_vmax, 9)
    exponent = -6
else:
    density_vmin, density_vmax = 0, 1.2e-3
    colorbar_ticks = np.linspace(density_vmin, density_vmax, 7)
    exponent = -3

for data_file in data_files:
    data = pd.read_csv(data_file)
    first_time = data['Time(s)'].min()
    first_time_cells =\
        data.groupby('CellID')['Time(s)'].min().eq(first_time).sum()
    groups = data.groupby('CellID')
    title = data_file.replace('_pos_export.txt', '')
        
    # Number of time points to evaluate the KDE. Include first and last time
    time_points = np.linspace(100, 1600, 4)
    time_points =\
        np.append(time_points, [min(data['Time(s)']), max(data['Time(s)'])])
        
    ###########################################################################
    # Plot 2D KDE
    ###########################################################################
    for time in time_points:
        x_positions, y_positions = [], []
        cell_count = 0
        
        for _, group in groups:
            x_group, y_group, time_group =\
                group['x(microns)'], group['y(microns)'], group['Time(s)']
            if (time < time_group.min() or
                time > time_group.max()):
                continue
            x_positions.append(np.interp(time, time_group, x_group))
            y_positions.append(np.interp(time, time_group, y_group))
            cell_count += 1
        
        x_positions, y_positions =\
            np.array(x_positions), 1250 - np.array(y_positions)
        
        grid_x = np.linspace(0, 1200, 200)
        grid_y = np.linspace(0, 1600, 200)
        X, Y = np.meshgrid(grid_x, grid_y)
        positions = np.vstack([y_positions, x_positions])
        
        if positions.shape[1] > 1:
            kde = gaussian_kde(positions,
                               bw_method=bandwidth_scaling*cell_count**(-1/6))
            density = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            if not div_cell_count:
                density *= cell_count
        else:
            density = np.zeros_like(X)
        
        plt.figure(figsize=(6, 8))
        levels = np.linspace(density_vmin, density_vmax, 25)
        cf = plt.contourf(X, Y, density, levels=levels, cmap="viridis",
                          vmin=density_vmin, vmax=density_vmax, extend='max')
        if plot_option in [1, 2]:
            plt.scatter(y_positions, x_positions, color='black', s=2)
        plt.xlim(0, 1200)
        plt.ylim(0, 1600)
        plt.xlabel("Position - $x$ $(\\mu m)$")
        plt.ylabel("Position - $y$ $(\\mu m)$")
        plt.grid()
        
        # Uncomment to see colourbar:
        # cbar = plt.colorbar(cf, 
            #ticks=colorbar_ticks, orientation='horizontal')
        # cbar.set_label(r'Density $( Cells / \mu m )$')
        
        plt.savefig(f"{title}_2D_{time:.1f}s_{div_cell_count}.png",
                        dpi=300, bbox_inches='tight')
        plt.show()
        
    ###########################################################################
    # Plot 1D KDE
    ###########################################################################
    plt.figure(figsize=(4, 4))
    colors = [cmap(i / (len(time_points) - 1)) for i in\
              range(len(time_points))]
    
    for idx, time in enumerate(time_points):
        new_x_positions = []
        cell_count = 0
        
        for _, group in groups:
            time_group, y_group = group['Time(s)'], group['y(microns)']
            if (time < time_group.min() or time >\
                time_group.max()):
                continue
            new_x_positions.append(1250 - np.interp(time, time_group, y_group))
            cell_count += 1
        
        new_x_positions = np.array(new_x_positions)
        grid_1d = np.linspace(0, 1200, 200)
        kde_1d = gaussian_kde(new_x_positions,
                              bw_method=bandwidth_scaling * cell_count**(-1/5))
        density_1d = kde_1d(grid_1d) * (first_time_cells if \
                                        rescaled else cell_count)
        plt.plot(grid_1d, density_1d, color=colors[idx], lw=5, alpha=0.7)
    
    plt.xlim(0, 1200)
    plt.xlabel("Position - $x$ $(\\mu m)$")
    plt.ylim(0, 1)
    plt.ylabel(r"Density $( Cells / \mu m )$")
    plt.grid()
    plt.savefig(f"{title}_1D.png", dpi=300, bbox_inches='tight')
    plt.show()
