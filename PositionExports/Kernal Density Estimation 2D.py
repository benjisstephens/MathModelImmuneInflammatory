# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 17:43:44 2024

@author: 44788
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% INITIALISE SETTINGS

# Start all paths to start at origin?
normalise = True  # True or False
bandwidth = 2

# Read the file names from 'all_data_set_names.txt'
with open("all_data_set_names.txt", "r") as f:
    data_files = f.read().splitlines()

#%% Process and plot each file individually

for data_file in data_files:
    # Load the CSV file
    data = pd.read_csv(data_file)
    
    # Remove '_pos_export.txt' from the file name for the plot title and saving
    title = data_file.replace('_pos_export.txt', '')
    output_file = f"{title}_2D_KDE_2.png"  # File name for saving

    if normalise:
        # Group the data by 'CellID'
        groups = data.groupby('CellID')

        # Ensure the columns are float before normalizing
        data['y(microns)'] = data['y(microns)'].astype(float)
        data['x(microns)'] = data['x(microns)'].astype(float)

        # Create normalized columns for y and x positions
        data['y_normal'] = 0.0
        data['x_normal'] = 0.0

        # Normalize the y and x values so that the first position starts at 0 for each cell
        for cell_id, group in groups:
            y_initial = group['y(microns)'].iloc[0]  # Initial y value for each cell
            data.loc[group.index, 'y_normal'] = (group['y(microns)'] - y_initial).astype(float)  # Subtract initial y
            
            x_initial = group['x(microns)'].iloc[0]  # Initial x value for each cell
            data.loc[group.index, 'x_normal'] = (group['x(microns)'] - x_initial).astype(float)  # Subtract initial x

        # Use normalized values
        data['y(microns)'] = data['y_normal']
        data['x(microns)'] = data['x_normal']

    # Plot individual dataset as a 2D contour plot
    plt.figure(figsize=(7, 6))

    # Create a 2D contour plot using KDE
    sns.kdeplot(data=data, x='x(microns)', y='y(microns)', bw_adjust=bandwidth, fill=True, cmap='viridis', levels=10)

    plt.xlabel('x (microns)')
    plt.ylabel('y (microns)')
    plt.title(f'2D Kernel Density Estimate for {title}')
    
    plt.axhline(y=0, linestyle= '--', color='gray')  # Horizontal line at y=0
    plt.axvline(x=0, linestyle= '--', color='gray')  # Vertical line at x=0
    
    # Set symmetrical limits for the x and y axes
    plt.xlim(-80, 80)  # Adjust as needed
    plt.ylim(-80, 80)  # Adjust as needed

    # Save the plot to a PNG file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save with high quality

    # Show the plot
    plt.show()
