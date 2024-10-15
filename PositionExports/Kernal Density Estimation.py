# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:51:57 2024

@author: 44788
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% INITIALISE SETTINGS

# Start all paths to start at origin?
normalise = True  # True or False
axis = 'x'  # 'x' or 'y'
bandwidth = 2

# Read the file names from 'all_data_set_names.txt'
with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

#%% Process and plot each file individually

for data_file in data_files:
    # Load the CSV file
    data = pd.read_csv(data_file)
    
    # Remove '_pos_export.txt' from the file name
    title = data_file.replace('_pos_export.txt', '')
    output_file = f"{title}_{axis}_1D_KDE.png"  # File name for saving

    if normalise:
        
        # Group the data by 'CellID'
        groups = data.groupby('CellID')

        if axis == 'y':
            # Ensure the y(microns) column is float before normalizing
            data['y(microns)'] = data['y(microns)'].astype(float)

            # Create a new column for normalized y positions, initialized with float type
            data['y_normal'] = 0.0

            # Normalize the y values so that the first position starts at 0 for each cell
            for cell_id, group in groups:
                y_initial = group['y(microns)'].iloc[0]  # Initial y value for each cell
                data.loc[group.index, 'y_normal'] = (group['y(microns)'] - y_initial).astype(float)  # Subtract initial y

            # Use normalized y values
            data['y(microns)'] = data['y_normal']

        elif axis == 'x':
            # Ensure the x(microns) column is float before normalizing
            data['x(microns)'] = data['x(microns)'].astype(float)

            # Create a new column for normalized x positions, initialized with float type
            data['x_normal'] = 0.0

            # Normalize the x values so that the first position starts at 0 for each cell
            for cell_id, group in groups:
                x_initial = group['x(microns)'].iloc[0]  # Initial x value for each cell
                data.loc[group.index, 'x_normal'] = (group['x(microns)'] - x_initial).astype(float)  # Subtract initial x

            # Use normalized x values
            data['x(microns)'] = data['x_normal']

    # Plot individual dataset
    plt.figure(figsize=(7, 6))

    if axis == 'y':
        sns.kdeplot(data['y(microns)'], bw_adjust=bandwidth, fill=True)
        plt.xlabel('y (microns)')
        plt.title(f'Kernel Density Estimate of y (microns) for {title}')
        
        # # Adjust limits of data so that 0 is central
        # y_min = data['y(microns)'].min()
        # y_max = data['y(microns)'].max()
        # limit = max(abs(y_min), abs(y_max))
        
        
    elif axis == 'x':
        sns.kdeplot(data['x(microns)'], bw_adjust=bandwidth, fill=True)
        plt.xlabel('x (microns)')
        plt.title(f'Kernel Density Estimate of x (microns) for {title}')
        
        # # Adjust limits of data so that 0 is central
        # x_min = data['x(microns)'].min()
        # x_max = data['x(microns)'].max()
        # limit = max(abs(x_min), abs(x_max))
       
    plt.xlim(-100, 100)  # Symmetrical limits around 0
    plt.ylim(0,0.09)
    plt.ylabel('Density')
    
    # Save the plot to a PNG file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high quality
     
    plt.show()

