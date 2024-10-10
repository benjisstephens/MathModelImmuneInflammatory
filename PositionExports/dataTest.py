# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:33:07 2024

@author: 44788
"""

import pandas as pd
import matplotlib.pyplot as plt

#%% INITIALISE SETTINGS

# Start all paths to start at origin?
normalise = False

#%%

# Load the CSV file
data = pd.read_csv('M1_DRC_CTRL_CCL19_50_I_pos_export.txt')

# Group by 'CellID'
groups = data.groupby('CellID')

# Create a plot
plt.figure(figsize=(10, 8))

# Plot each cell's trajectory
for cell_id, group in groups:
    
    
    # Set normalise = False to show relative starting positions of cells.
    if normalise == False:
        x_group = group['x(microns)'] 
        y_group = group['y(microns)']
    
    # Set normalise = True to normalise each cell path to start at origin.
    if normalise == True:
        x_group = group['x(microns)'] - group['x(microns)'].iloc[0]
        y_group = group['y(microns)'] - group['y(microns)'].iloc[0]
    
    plt.plot(x_group, y_group)

# Add labels and a legend
plt.xlabel('x (microns)')
plt.ylabel('y (microns)')
plt.title('Cell Trajectories')
plt.legend()

# Show the plot
plt.show()

