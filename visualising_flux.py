import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import gaussian
import matplotlib.ticker as mticker

# %% Define parameters
plt.rcParams.update({'font.size': 14})

# Define spatial boundaries
x0 = 0
x1 = 1200
dx = 100
Dx = np.linspace(x0, x1, dx + 1)

# Define temporal boundaries
t0 = 2.8
t1 = 1798
dt = 100
Dt = np.linspace(t0, t1, dt + 1)

# %% Get flux data

# Function to determine the band index for a given position
def get_band_index(y_pos, boundaries):
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= y_pos < boundaries[i + 1]:
            return i
    return None

# Store all data in a dictionary
flux_data = {}

# Iterate through all data sets
with open('all_data_set_names.txt', 'r') as file:
    data_files = [line.strip() for line in file.readlines()]

for file_path in data_files:
    # Initialise flux_over_time to be 0 for each time bin and spatial band.
    flux_over_time = {}
    for t in range(len(Dt) - 1):
        flux_over_time[t] = {}
        for i in range(len(Dx)):
            flux_over_time[t][i] = 0

    print("Processing:", file_path)
    df = pd.read_csv(file_path)

    # Transform data
    df['y_transformed'] = 1250 - df['y(microns)']
    df = df[(df['y_transformed'] > 0) & (df['y_transformed'] < 1200)]
    df['Band'] = df['y_transformed'].apply(get_band_index, boundaries=Dx)

    first_time = df['Time(s)'].min()
    initial_cells = df.groupby('CellID')['Time(s)'].min().eq(first_time).sum()

    # Count the cells in each time bin.
    cell_count_over_time = {}
    for t in range(len(Dt) - 1):
        current_cells = df[(df['Time(s)'] >=\
                    Dt[t]) & (df['Time(s)'] < Dt[t + 1])]['CellID'].nunique()
        cell_count_over_time[t] = current_cells

    # Count boundary crossing events.
    for cell_id in df['CellID'].unique():
        cell_data = df[df['CellID'] == cell_id].sort_values(by='Time(s)')

        # Check whether a cell has moved across a boundary in the position.
        for i in range(1, len(cell_data)):
            prev_time = cell_data.iloc[i - 1]['Time(s)']
            current_time = cell_data.iloc[i]['Time(s)']
            prev_band = cell_data.iloc[i - 1]['Band']
            current_band = cell_data.iloc[i]['Band']

            if prev_band is not None and current_band is not None and \
                prev_band != current_band:
                # Find the time bin in which the event occurred 
                for t in range(len(Dt) - 1):
                    if Dt[t] <= prev_time < Dt[t + 1]:
                        # For an upward crossing 
                        if prev_band < current_band:
                            flux_over_time[t][prev_band + 1] += 1
                        # For a downward crossing
                        else:
                            flux_over_time[t][current_band + 1] -= 1
                        break 

    # Normalize the flux counts in each time bin.
    for t in range(len(Dt) - 1):
        if cell_count_over_time[t] > 0:
            norm_factor = initial_cells / cell_count_over_time[t]
        else:
            norm_factor = 1 
        for i in range(len(Dx)):
            flux_over_time[t][i] *= norm_factor

    flux_matrix = []
    for t in range(len(Dt) - 1):
        row = []
        for i in range(len(Dx)):
            row.append(flux_over_time[t][i])
        flux_matrix.append(row)
        
    flux_matrix = np.array(flux_matrix)
    flux_matrix = np.clip(flux_matrix, -5, 5)
    flux_matrix /= (dt)
    flux_data[file_path] = flux_matrix
    
# Generate exact plots
vlim = 3e-2

for file_path, flux_matrix in flux_data.items():
    base_title = file_path.replace('_pos_export.txt', '')
    
    fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7), 
                                      gridspec_kw={"height_ratios": [1, 20]}, 
                                      dpi=300)
    cmap = plt.get_cmap('seismic')
    vmin, vmax = -vlim, vlim
    norm = mcolors.Normalize(vmin, vmax)
    pcm = ax.pcolormesh(Dx, Dt[:-1], flux_matrix, shading='nearest', 
                        cmap=cmap, norm=norm)
    ax.set_xlabel('Position $(\\mu m)$')
    ax.set_ylabel('Time (s)')
    ax.set_yticks([0, 300, 600, 900, 1200, 1500, 1800])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', 
                        location='top')
    cbar.set_label('Flux $(cells/s)$', fontsize=14)
    formatter = mticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar.ax.xaxis.set_major_formatter(formatter)
    cbar.set_ticks(np.linspace(vmin, vmax, 3))
    title = base_title
    plt.tight_layout()
    plt.savefig(f"{title}_Flux_Contour.png", dpi=300, bbox_inches='tight')
    plt.show()

# Generate smoothed plots
for bandwidth in [5, 10]:
    if bandwidth == 5:
        vlim = 1e-3
    else:
        vlim = 2e-4

    for file_path, flux_matrix in flux_data.items():
        base_title = file_path.replace('_pos_export.txt', '')

        flux_matrix_smoothed = gaussian_filter(flux_matrix.astype(float), 
                                               sigma=bandwidth)

        size = int(6 * bandwidth)
        gaussian_kernel = gaussian(size, bandwidth)
        kernel_sum = np.sum(gaussian_kernel)
        flux_matrix_smoothed_normalized = flux_matrix_smoothed / kernel_sum
        np.savetxt(f"{base_title}_flux_data.txt",
                   flux_matrix_smoothed_normalized, delimiter=",")
        

        fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7), 
            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)
        cmap = plt.get_cmap('seismic')
        vmin, vmax = -vlim, vlim
        norm = mcolors.Normalize(vmin, vmax)
        pcm = ax.pcolormesh(Dx, Dt[:-1], flux_matrix_smoothed_normalized, 
                            shading='nearest', cmap=cmap, norm=norm)
        ax.set_xlabel('Position $(\\mu m)$')
        ax.set_ylabel('Time (s)')
        ax.set_yticks([0, 300, 600, 900, 1200, 1500, 1800])
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', 
                            location='top')
        cbar.set_label('Flux $(cells/s)$', fontsize=14)
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.xaxis.set_major_formatter(formatter)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        title = base_title
        plt.tight_layout()
        plt.savefig(f"{title}_Flux_Contour_{bandwidth}.png", dpi=300, 
                    bbox_inches='tight')
        plt.show()
