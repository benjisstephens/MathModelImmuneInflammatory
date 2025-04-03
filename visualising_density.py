import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde  # Use SciPy’s KDE
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker

# %% INITIALISE SETTINGS
plt.rcParams.update({'font.size': 14})
plt.minorticks_on()

# Read the file names from 'all_data_set_names.txt'
with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

extrapolate = False

largest_max_density = 0  
largest_density_file = None
largest_density_time = None   
largest_density_position = None
        
for data_file in data_files:
    data = pd.read_csv(data_file)
    
    # Compute first_time_cells exactly as in Code 2 (div_cell_count=True, rescaled=True)
    first_time = data['Time(s)'].min()
    first_time_cells = data.groupby('CellID')['Time(s)'].min().eq(first_time).sum()
    
    # Create time points; include the min and max times as in Code 1.
    time_points = np.linspace(3, 1797, 98)
    time_points = np.append(time_points, max(data['Time(s)']))
    time_points = np.append(min(data['Time(s)']), time_points)
    
    groups = data.groupby('CellID')
    
    # Collect the transformed y positions for each time step.
    all_positions = []  # each element is a list of positions (transformed from y)
    time_values = []    # times for which we have positions
    
    for time in time_points:
        y_positions = []
        for cell_id, group in groups:
            x_group = group['x(microns)']
            y_group = group['y(microns)']
            time_group = group['Time(s)']
    
            # Only consider times within the available range for the cell.
            if (time < time_group.min() or time > time_group.max()) and not extrapolate:
                continue
    
            # Interpolate y position and transform (flip y-axis as before)
            y_pos = np.interp(time, time_group, y_group)
            y_positions.append(1250 - y_pos)
    
        all_positions.append(y_positions)
        if y_positions:
            time_values.append(time)
    
    # Define the grid for evaluating the KDE.
    grid_x = np.linspace(0, 1200, 101)  # 200 points to be similar to Code 2
    grid_t = time_points
    density = np.zeros((len(grid_t), len(grid_x)))
    
    # For each time step compute the KDE using gaussian_kde.
    for i, y_positions in enumerate(all_positions):
        if len(y_positions) > 1:
            cell_count = len(y_positions)
            
            # Compute KDE with proper bandwidth scaling
            kde = gaussian_kde(y_positions, bw_method=cell_count**(-1/5))
            
            # Evaluate KDE on grid
            density_vals = kde(grid_x)
            
            # Apply the correct scaling
            density_vals *= first_time_cells  # Matches rescaled=True in Code 2
    
            density[i, :] = density_vals
    
            # Track the maximum density
            max_density = density_vals.max()
            if max_density > largest_max_density:
                largest_max_density = max_density
                largest_density_file = data_file
                largest_density_time = grid_t[i]
                largest_density_position = grid_x[np.argmax(density_vals)]
                
        else:
            density[i, :] = np.zeros_like(grid_x)
            
        
    print(np.shape(density))
        
    title = data_file.replace('_pos_export.txt', '')
    np.savetxt(f"{title}_density_data.txt", density, delimiter=",")

    fig, (cbar_ax, ax) = plt.subplots(nrows=2, 
            figsize=(6, 7), gridspec_kw={"height_ratios": [1, 20]}, dpi=300
    )
    cmap = plt.get_cmap('magma')
    vmin, vmax = 0, 1
    norm = mcolors.Normalize(vmin, vmax)
    pcm = ax.pcolormesh(grid_x, grid_t, density, shading='auto', cmap=cmap, norm=norm)
    ax.set_xlabel('Position $(\\mu m)$')
    ax.set_ylabel('Time (s)')
    ax.set_yticks([0, 300, 600, 900, 1200, 1500, 1800])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', location='top')
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Force scientific notation (e.g., 1e0)
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.set_label('Density $(cells/\mu m)$', fontsize=14)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    plt.tight_layout(pad=1.0)  # Adjust padding to ensure proper spacing
    plt.savefig(f"{title}_density_contour.png", dpi=300, bbox_inches='tight')
    plt.show()


