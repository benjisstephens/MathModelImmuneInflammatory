import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% INITIALISE SETTINGS

# Start all paths to start at origin?
normalise = True

#%%

# Load the CSV file
data = pd.read_csv('M1_DRC_CTRL_CCL19_50_I_pos_export.txt')

plt.figure(figsize=(10, 6))

if normalise == True:
    
    # Group the data by 'CellID'
    groups = data.groupby('CellID')
    
    # Create a new column for normalized y positions
    data['y_normal'] = 0
    
    # Normalize the y values so that the first position starts at 0 for each cell
    for cell_id, group in groups:
        y_initial = group['y(microns)'].iloc[0]  # Initial y value for each cell
        data.loc[group.index, 'y_normal'] = group['y(microns)'] - y_initial  # Subtract initial y
    
    # Retitle the data column as it was before editing
    data['y(microns)'] = data['y_normal']
    
# Add labels and a title
sns.kdeplot(data['y(microns)'], bw_adjust=1, fill=True)
plt.xlabel('y (microns)')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of y (microns) for Cell Positions')

# Show the plot
plt.show()

