# Code to plot cell trajectory data for all datasets.
# - Raw cell trajectories
# - Transformed cell trajectories
# - Distribution of the time cells are tracked
# - Cell count over time
# - Distribution of displacement of cells 
#      (relative to the time they are tracked)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({'font.size': 14})

normalise = False

with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

for data_file in data_files:
    data = pd.read_csv(data_file)
    groups = data.groupby('CellID')
    title = data_file.replace('_pos_export.txt', '')

    ###########################################################################
    # Plot raw cell trajectories
    ###########################################################################
    plt.figure(figsize=(8, 6))
    for cell_id, group in groups:
        x_group = group['x(microns)'] -\
            (group['x(microns)'].iloc[0] if normalise else 0)
        y_group = group['y(microns)'] -\
            (group['y(microns)'].iloc[0] if normalise else 0)
        plt.plot(x_group, y_group, label=f'Cell {cell_id}')
    plt.xlim(-50, 1700)
    plt.ylim(-100, 1300)
    plt.xlabel(r'Position - $x$ ($\mu m$)')
    plt.ylabel(r'Position - $y$ ($\mu m$)')
    plt.grid()
    plt.savefig(f"{title}_Raw_Cell_Trajectories.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    ###########################################################################
    # Plot rotated cell trajectories
    ###########################################################################
    plt.figure(figsize=(6, 8))
    for cell_id, group in groups:
        x_group = group['x(microns)'] -\
            (group['x(microns)'].iloc[0] if normalise else 0)
        y_group = 1250 - group['y(microns)'] -\
            (group['y(microns)'].iloc[0] if normalise else 0)
        valid_indices = (y_group >= 0) & (y_group <= 1200)
        plt.plot(y_group[valid_indices], x_group[valid_indices],
                 label=f'Cell {cell_id}')
    plt.xlim(0, 1200)
    plt.ylim(0, 1600)
    plt.xlabel(r'Position - $x$ ($\mu m$)')
    plt.ylabel(r'Position - $y$ ($\mu m$)')
    plt.grid()
    plt.savefig(f"{title}_Cell_Trajectories.png", dpi=300, bbox_inches='tight')
    plt.show()

    ###########################################################################
    # Calculate cell count over time
    ###########################################################################
    cell_appearance_times =\
        data.groupby('CellID')['Time(s)'].agg(['min', 'max'])
    time_steps = np.sort(data['Time(s)'].unique())
    cell_count_per_time = {time: ((cell_appearance_times['min'] <= time) &\
        (cell_appearance_times['max'] >= time)).sum() for time in time_steps}
    print(f"Number of cells visible at the first time ({time_steps[0]}s):",
          cell_count_per_time[time_steps[0]])
    
    ###########################################################################
    # Plot tracking duration histogram
    ###########################################################################
    durations = (cell_appearance_times['max'] -\
                 cell_appearance_times['min']).values
    plt.figure(figsize=(6, 3))
    plt.hist(durations, bins=np.arange(0, max(durations) + 10, 100),
             color='darkblue', edgecolor='black', alpha=0.7, zorder=3)
    plt.xlabel('Time a Cell is Tracked (s)')
    plt.ylabel('Number of Cells')
    plt.xlim(0, 1800)
    plt.xticks(np.linspace(0, 1800, 7))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{title}_tracking_duration_histogram.png", dpi=300,
                bbox_inches='tight')
    plt.show()

    ###########################################################################
    # Plot cell count over time
    ###########################################################################
    plt.figure(figsize=(6, 2))
    plt.plot(list(cell_count_per_time.keys()),
             list(cell_count_per_time.values()), color='darkgreen')
    plt.yscale("log")
    plt.xlim([0, 1800])
    plt.ylim([10, 2000])
    plt.xticks([0, 300, 600, 900, 1200, 1500, 1800])
    plt.yticks([10, 100, 1000])
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                             pos: '%d' % x))
    plt.xlabel('Time (s)')
    plt.ylabel('log( Cell Count )')
    plt.grid()
    plt.savefig(f"{title}_Cell_Count.png", dpi=300, bbox_inches='tight')
    plt.show()

    ###########################################################################
    # Cell displacement per 30 minutes
    ###########################################################################
    disp = []
    for cell_id, group in groups:
        initial_x, initial_y = group.iloc[0][['x(microns)', 'y(microns)']]
        final_x, final_y = group.iloc[-1][['x(microns)', 'y(microns)']]
        displacement = np.sqrt((final_x - initial_x)**2 + 
                               (final_y - initial_y)**2)
        duration = group['Time(s)'].max() - group['Time(s)'].min()
        if duration > 0:
            disp.append((displacement / duration) * 1800)
    disp = np.array(disp)
    percentiles = np.percentile(disp, [50, 75, 80, 90, 95])
    print(f"{title}")
    print("Bound proportion:")
    print("{1 - (np.sum(np.array(displacement_per_30min) < 50.40) / len(displacement_per_30min)):.2f}")
    print("Percentile of displacement distribution:")
    for i, p in enumerate([50, 75, 80, 90, 95]):
        print(f"{p}th: {percentiles[i]:.2f}")

    ###########################################################################
    # Plot displacement histogram
    ###########################################################################
    bins = np.append(np.arange(0, 300, 10), np.inf)
    counts, bin_edges = np.histogram(disp, bins=bins)
    plt.figure(figsize=(6, 3))
    plt.bar(bin_edges[:-2], counts[:-1], width=10, color='darkgreen', 
            edgecolor='black', alpha=0.7, align='edge', zorder=3)
    plt.bar(bin_edges[-2], counts[-1], width=10, color='darkgrey',
            edgecolor='black', alpha=0.7, align='edge', zorder=3)
    plt.xlabel('Displacement $(\mu m)$ per 30 minutes')
    plt.ylabel('Number of Cells')
    plt.xlim(0, 300)
    plt.xticks(np.arange(0, 320, 30))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{title}_displacement_per_time_histogram.png",
                dpi=300, bbox_inches='tight')
    plt.show()
