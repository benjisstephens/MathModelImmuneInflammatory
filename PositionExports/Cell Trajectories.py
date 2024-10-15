# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:33:07 2024

@author: 44788
"""

import pandas as pd
import matplotlib.pyplot as plt

#%% INITIALISE SETTINGS

# Start all paths to start at origin?
normalise = True  # Set to True if you want to normalize the paths to start at the origin

# Read the file names from 'all_data_set_names.txt'
with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

#%% Process and plot each file individually

for data_file in data_files:
    # Load the CSV file
    data = pd.read_csv(data_file)
    
    # Group by 'CellID'
    groups = data.groupby('CellID')

    plt.figure(figsize=(7, 6))

    for cell_id, group in groups:

        # Set normalise = False to show relative starting positions of cells
        if normalise == False:
            x_group = group['x(microns)']
            y_group = group['y(microns)']

        # Set normalise = True to normalize each cell path to start at origin
        if normalise == True:
            x_group = group['x(microns)'] - group['x(microns)'].iloc[0]
            y_group = group['y(microns)'] - group['y(microns)'].iloc[0]

        plt.plot(x_group, y_group, label=f'Cell {cell_id}')

    # Set labels and title
    plt.xlabel('x (microns)')
    plt.ylabel('y (microns)')
    
    plt.xlim(-125, 125)  # Adjust as needed
    plt.ylim(-125, 125)  # Adjust as needed
    
    # Remove '_pos_export.txt' from the file name for the plot title and save file name
    title = data_file.replace('_pos_export.txt', '')
    plt.title(f'Cell Trajectories for {title}')
    
    # Add grid
    plt.grid()

    # Save the plot as an image file
    plt.savefig(f"{title}_Cell_Trajectories.png", dpi=300, bbox_inches='tight')

    # Show the plot (optional, can be removed if only saving)
    plt.show()

