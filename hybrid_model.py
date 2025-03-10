# Code to simulate a hybrid model
# - runs simulation
# - collect density and flux data
# - visualise desnity distribution
# - visualise flux distribution

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.ticker as mticker
from scipy.signal.windows import gaussian

plt.rcParams.update({'font.size': 14})

# Select which case to examine
chemokine_present = True
flow_direction = 'NEG'

threeDPlot = False  # True: display both 2D and 3D views in animation;
                    # False: display only 2D in animation. 
                   
# Initalise constants based on experimental condition
if chemokine_present == False: 
    data_file = "M4_wDC_CTRL_POS_pos_export.txt" 
    U = -2 * 8.4e-2                 # um s^-1     
    proportion_phi_c = 0
else:
    if flow_direction == 'NEG':
        U = 2 * 8.4e-2               # um s^-1 
        data_file = "M12_wDC_CCL21_NEG_pos_export.txt"
        proportion_phi_c = 0.2
    elif flow_direction == 'DIF':
        data_file = "M12_wDC_CCL21_DIF_pos_export.txt"
        U = 0                        # um s^-1   
        proportion_phi_c = 0.36
    elif flow_direction == 'POS':
        data_file = "M12_wDC_CCL21_POS_pos_export.txt"
        U = 2 * 8.4e-2               # um s^-1   
        proportion_phi_c = 0.36

# Domain boundaries
left_boundary = 0
right_boundary = 1200
lower_boundary = 0
upper_boundary = 1600

# Animation parameters
total_simulation_time = 1800
frames_per_second = 18

# Stochastic difference equation parameters
D = 0.41            # Diffusion parameter in um^2 s^-1
chi = 10            # Chemotactic parameter in um^2 s^-1 cells^-1 um^3
k_N7_p = 1.83e-3    # Rate phi -> phi_c in s^-1 cells^-1 um^3
k_N7_m = 5e-3       # Rate phi_c -> phi in s^-1

# Function which alters the chemokine concentration after binding/unbinding
def change_chemokine_gradient(x, y, x0, y0):
    return 0.048 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * 100**2))

# Class which defines a cell
class Cell:
    def __init__(self, x, y, radius, is_bound=False):
        self.r = np.array((x, y))   # Cell position
        self.radius = radius        # Cell radius
        self.is_bound = is_bound    # Flag if cell is 'bound' to chemokine
        # Change colour of cell if 'bound'
        if self.is_bound: 
            self.circle = Circle(xy=self.r, radius=self.radius, 
                                 edgecolor='red',facecolor='red',fill=True)
        else: 
            self.circle = Circle(xy=self.r, radius=self.radius, 
                                 edgecolor='green',facecolor='green',fill=True)
        
        # Initalise cell positions based on intial positions in data
        self.initial_position = np.array((x, y)) 
        
    @property
    def x(self):
        return self.r[0]
    
    @x.setter
    def x(self, value):
        self.r[0] = value
        
    @property
    def y(self):
        return self.r[1]
    
    @y.setter
    def y(self, value):
        self.r[1] = value
        
    # Ensure cells do no overlap
    def check_for_overlap(self, other):
        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    # Plot cell positions
    def draw(self, ax):
        ax.add_patch(self.circle)
        return self.circle

    # Advance the cells at each time step
    def advance_cell(self, dt, diffusion_coeff, concentration, gradient_func):
        # Move randomly by diffusion (brownian motion)
        proposed_r = self.r + np.sqrt(2*diffusion_coeff*dt)*np.random.randn(2)
    
        # Move chemotactically (up gradient of c) if 'bound'
        if self.is_bound:
            proposed_r += chi*gradient_func(self.x, self.y, concentration)*dt
    
        # Cells cannot cross border boundaries
        if left_boundary + self.radius <= proposed_r[0] <= right_boundary \
            - self.radius and \
           lower_boundary + self.radius <= proposed_r[1] <= upper_boundary \
            - self.radius:
            self.r = proposed_r 
        else:
            self.r = proposed_r 

    # Define function for binding/unbinding events
    def update_type(self, binding_rate, unbinding_rate, concentration):
        
        # Binding: Phi -> Phi_c
        if not self.is_bound:
            if np.random.rand() < binding_rate:
                self.is_bound = True
                self.modify_concentration(concentration, is_binding=True)
                self.circle.set_edgecolor('red')
                self.circle.set_facecolor('red')
                
        # Unbinding: Phi_c -> Phi
        elif self.is_bound:
            if np.random.rand() < unbinding_rate:
                self.is_bound = False
                self.modify_concentration(concentration, is_binding=False)
                self.circle.set_edgecolor('green')
                self.circle.set_facecolor('green')

    # When binding/unbinding occurs, change concentration of c
    def modify_concentration(self, concentration, is_binding):
        x, y = self.x, self.y
        x_indices = np.linspace(left_boundary, right_boundary, 
                                concentration.shape[1])
        y_indices = np.linspace(lower_boundary, upper_boundary, 
                                concentration.shape[0])
        X, Y = np.meshgrid(x_indices, y_indices)
        gaussian = change_chemokine_gradient(X, Y, x, y)
        # Increase c when unbinding occurs, decrease c when binding occurs
        if is_binding:
            concentration -= gaussian
        else:
            concentration += gaussian
        # Fix concentration as positive
        concentration = np.clip(concentration, 0, None)

# Class which defines a simulation
class Simulation:
    # Initialise data
    def __init__(self, radius=6):
        self.initial_positions = {}
        self.init_cells(radius)
        self.initialise_concentration()
        self.boundaries = np.linspace(left_boundary, right_boundary, 100)[1:-1]
        self.boundary_crossings = {b: 0 for b in self.boundaries}
        self.num_frames = int(total_simulation_time / frames_per_second)
        self.density_matrix = np.zeros((self.num_frames, 100))
        self.final_positions = {}
        self.msd_values = []

    # Initalise cells based on data
    def init_cells(self, radius):
        
        # Initialise cell positions based on data
        self.cells = []
        df = pd.read_csv(data_file)
        df_filtered = df[df['Time(s)'].round(1) == 2.8]
        df_shuffled = df_filtered.sample(frac=1,
                                        random_state=42).reset_index(drop=True)
        proportion_bound = len(df_shuffled) * (proportion_phi_c)
        for i, row in df_shuffled.iterrows():
            x, y = row['x(microns)'], row['y(microns)']
            transformed_x, transformed_y = 1250 - y, x
            is_bound = i < proportion_bound
            cell = Cell(transformed_x, transformed_y, radius,is_bound=is_bound)
            self.cells.append(cell)
            for cell in self.cells:
                self.initial_positions[cell] = np.array(cell.r)

    # Update each frame of the simulation 
    def advance_simulation(self, dt, diffusion_coeff):
        previous_positions = {cell: cell.x for cell in self.cells}
        
        for cell in self.cells:
            cell.advance_cell(dt, diffusion_coeff, self.c, 
                                                  self.gradient_concentration)
            local_concentration = self.get_local_concentration(cell.x, cell.y)
            cell.update_type(k_N7_p*local_concentration*dt, k_N7_m*dt, self.c)
    
        for cell in self.cells:
            self.final_positions[cell] = np.array(cell.r)
    
        # Update the mean squared displacement for cells  
        #   Used to check diffusion matches at micro and macro scales
        msd, std_msd = self.calculate_msd()
        self.msd_values.append((msd, std_msd))
    
        # Count boundary crossings for flux calculations
        time_step_crossings = {boundary: 0 for boundary in self.boundaries}
        for cell in self.cells:
            previous_x = previous_positions[cell]
            for boundary in self.boundaries:
                if previous_x < boundary <= cell.x:
                    self.boundary_crossings[boundary] += 1
                    time_step_crossings[boundary] += 1
                elif previous_x > boundary >= cell.x:
                    self.boundary_crossings[boundary] -= 1
                    time_step_crossings[boundary] -= 1
    
        # Smooth chemokine field while preserving boundaries
        self.c = gaussian_filter(self.c, sigma=1)
        if chemokine_present:
            self.c[:, 0] = 6.022  
        else:
            self.c[:, 0] = 0
        self.c[:, -1] = 0
    
        # Return flux data
        return time_step_crossings
    
    # Calculate mean squared displacement 
    def calculate_msd(self):
        squared_displacements = []
        for cell in self.cells:
            initial_pos = self.initial_positions[cell]
            final_pos = self.final_positions[cell]
            displacement = np.linalg.norm(final_pos - initial_pos)
            squared_displacements.append(displacement**2)
        mean_squared_displacement = np.mean(squared_displacements)
        std_squared_displacement = np.std(squared_displacements)
        return mean_squared_displacement, std_squared_displacement

    # Initalise the chemokine concentration
    def initialise_concentration(self):
        x = np.linspace(left_boundary, right_boundary, 100)
        y = np.linspace(lower_boundary, upper_boundary, 100)
        D_val = 100         # um^2 s^-1
        
        if chemokine_present == True:
            C = 6.022           
        else: 
            C = 0
        d = 1.3e-6      
        M1 = (U + np.sqrt(U**2 + 4 * D_val * d)) / (2 * D_val)
        M2 = (U - np.sqrt(U**2 + 4 * D_val * d)) / (2 * D_val)
        A = C / (1 - np.exp((M1 - M2) * right_boundary))
        self.X, self.Y = np.meshgrid(x, y)
        self.c = A * np.exp(M1 * self.X) + (C - A) * np.exp(M2 * self.X)
        self.c[:, 0] = C
        self.c[:, -1] = 0  
        self.c = gaussian_filter(self.c, sigma=1)

    # Approximate the local chemokine concentration at a given (x,y)
    def gradient_concentration(self, x, y, c_field):
        dx = (c_field.shape[1] - 1) / (right_boundary - left_boundary)
        dy = (c_field.shape[0] - 1) / (upper_boundary - lower_boundary)
        i = int((y - lower_boundary) * dy)
        j = int((x - left_boundary) * dx)
        i = np.clip(i, 1, c_field.shape[0] - 2)
        j = np.clip(j, 1, c_field.shape[1] - 2)
        grad_x = (c_field[i, j + 1] - c_field[i, j - 1]) / (2 / dx)
        grad_y = (c_field[i + 1, j] - c_field[i - 1, j]) / (2 / dy)
        return np.array([grad_x, grad_y])

    # Approximate the local chemokine concentration at a given (x,y)
    def get_local_concentration(self, x, y):
        dx = (self.c.shape[1] - 1) / (right_boundary - left_boundary)
        dy = (self.c.shape[0] - 1) / (upper_boundary - lower_boundary)
        ix = int(np.clip((x - left_boundary) * dx, 0, self.c.shape[1] - 1))
        iy = int(np.clip((y - lower_boundary) * dy, 0, self.c.shape[0] - 1))
        return self.c[iy, ix]
    
    # Calculate the density of cells
    def calculate_kde(self, x_positions, cell_count):
        kde = gaussian_kde(x_positions, bw_method=1 * cell_count**(-1/5))
        x_grid = np.linspace(0, 1200, 100)
        density = kde(x_grid) * cell_count
        return x_grid, density

    # Plot the density data
    def plot_density_matrix(self):
        fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7),
                            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)
        time = np.linspace(0, total_simulation_time, self.num_frames)
        x_grid = np.linspace(left_boundary, right_boundary, 100)
        vmin, vmax = 0, 1
        im = ax.pcolormesh(x_grid, time, self.density_matrix, shading='auto',
                           cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Position $(\mu m)$')
        ax.set_ylabel('Time (s)')
        ax.set_xticks(np.linspace(left_boundary, right_boundary, 7))
        ax.set_yticks(np.linspace(0, total_simulation_time, 7))
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            location='top')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Density ($cells/\mu m$)', fontsize=14)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_powerlimits((0, 0))  
        cbar.ax.xaxis.set_major_formatter(formatter)
        plt.tight_layout(pad=1.0)
        plt.savefig("Density_Matrix.png", dpi=300, bbox_inches='tight')
        plt.show()

    # Plot the flux data
    def plot_flux_matrix(self, flux_matrix):
        x, y = np.meshgrid(np.arange(flux_matrix.shape[1]), 
                           np.arange(flux_matrix.shape[0]))
        x_scaled = x * (right_boundary-left_boundary)/(flux_matrix.shape[1]-1)
        y_scaled = y * total_simulation_time / self.num_frames
    
        # Define normalization limits for raw flux data
        vmin_raw, vmax_raw = -3e-2, 3e-2
        norm_raw = mcolors.Normalize(vmin=vmin_raw, vmax=vmax_raw)
    
        # Create figure with colorbar for raw flux plot
        fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7), 
                            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)
        flux_plot = ax.pcolormesh(x_scaled, y_scaled, flux_matrix,
                            cmap='seismic', norm=norm_raw, shading='nearest')
    
        # Configure axes
        ax.set_xlabel('Position $(\mu m)$')
        ax.set_ylabel('Time (s)')
        ax.set_xticks(np.arange(left_boundary, right_boundary + 1, 200))
        ax.set_yticks(np.arange(0, total_simulation_time + 1, 300))
    
        # Create and configure colorbar
        cbar = fig.colorbar(flux_plot, cax=cbar_ax, orientation='horizontal',
                            location='top')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Flux', fontsize=14)
        cbar.set_ticks(np.linspace(vmin_raw, vmax_raw, 5))
    
        # Format colorbar tick labels
        formatter = mticker.ScalarFormatter(useMathText=False)
        formatter.set_powerlimits((0, 0))
        cbar.ax.xaxis.set_major_formatter(formatter)
    
        # Adjust layout and save
        plt.tight_layout(pad=1.0)
        plt.savefig("Flux_Matrix_Raw.png", dpi=300, bbox_inches='tight')
        plt.show()
    
        # Loop over smoothing bandwidths and create colorbars
        for bandwidth in [1,3,5]:
            fig, (cbar_ax, ax) = plt.subplots(nrows=2, figsize=(6, 7), 
                            gridspec_kw={"height_ratios": [1, 20]}, dpi=300)
    
            # Apply Gaussian smoothing
            flux_matrix_smoothed = gaussian_filter(flux_matrix.astype(float),
                                                   sigma=bandwidth)
    
            # Normalize using Gaussian kernel sum
            size = int(6 * bandwidth)
            gaussian_kernel = gaussian(size, bandwidth)
            kernel_sum = np.sum(gaussian_kernel)
            flux_matrix_smoothed_normalized = flux_matrix_smoothed / kernel_sum
            
            np.savetxt("flux_matrix.txt", flux_matrix_smoothed_normalized,
                       delimiter=",")
    
            # Define normalization limits for smoothed flux data
            vmin_smooth, vmax_smooth = -1e-3, 1e-3
            norm_smooth = mcolors.Normalize(vmin=vmin_smooth, vmax=vmax_smooth)
    
            # Create smoothed flux plot
            flux_plot_smoothed = ax.pcolormesh(x_scaled, y_scaled, 
                flux_matrix_smoothed_normalized, cmap='seismic', 
                norm=norm_smooth, shading='gouraud')
    
            # Configure axes
            ax.set_xlabel('Position $(\mu m)$')
            ax.set_ylabel('Time (s)')
            ax.set_xticks(np.arange(left_boundary, right_boundary + 1, 200))
            ax.set_yticks(np.arange(0, total_simulation_time + 1, 300))
    
            # Create and configure colorbar
            cbar = fig.colorbar(flux_plot_smoothed, cax=cbar_ax, 
                                orientation='horizontal', location='top')
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Flux (Smoothed)', fontsize=14)
            cbar.set_ticks(np.linspace(vmin_smooth, vmax_smooth, 5))
    
            # Format colorbar tick labels
            cbar.ax.xaxis.set_major_formatter(formatter)
    
            # Adjust layout and save
            plt.tight_layout(pad=1.0)
            plt.savefig(f"Flux_Matrix_Smoothed_{bandwidth}.png", dpi=300, 
                        bbox_inches='tight')
            plt.show()
     
    # Output the mean squared deviation of all cells
    def print_final_msd(self):
        if self.msd_values:
            final_msd, final_std = self.msd_values[-1]
            print(f"Final MSD: {final_msd:.4f}")
            print(f"Final std. dev. of squared displacement: {final_std:.4f}")
      
    # Do the animation
    def do_animation(self, total_time, num_frames, save=True):
        # Plot two subfigures: the cells and the chemokine gradient
        if threeDPlot:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_aspect('equal', 'box')
            ax1.set_xlim(left_boundary, right_boundary)
            ax1.set_ylim(lower_boundary, upper_boundary)
            ax1.set_xlabel("x (microns)")
            ax1.set_ylabel("y (microns)")
            ax1.set_xticks(np.arange(left_boundary, right_boundary + 1, 200))
            ax1.set_yticks(np.arange(lower_boundary, upper_boundary + 1, 200))
            ax1.tick_params(axis='both', which='major')
            
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            ax2.set_xlim(left_boundary, right_boundary)
            ax2.set_ylim(lower_boundary, upper_boundary)
            ax2.set_xticks(np.arange(left_boundary, right_boundary + 1, 400))
            ax2.set_yticks(np.arange(lower_boundary, upper_boundary + 1, 400))
            ax2.set_zlim(0, 6.5)
            ax2.set_xlabel("x (microns)")
            ax2.set_ylabel("y (microns)")
            ax2.set_zlabel("Concentration C")
            ax2.set_title("3D View of Concentration C")
            
            plt.show()
        # Plot one figure: just the cells
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.set_aspect('equal', 'box')
            ax1.set_xlim(left_boundary, right_boundary)
            ax1.set_ylim(lower_boundary, upper_boundary)
            ax1.set_xlabel("x (microns)")
            ax1.set_ylabel("y (microns)")
            ax1.set_xticks(np.arange(left_boundary, right_boundary + 1, 200))
            ax1.set_yticks(np.arange(lower_boundary, upper_boundary + 1, 200))
        
        concentration_plot = ax1.imshow(self.c, extent=[left_boundary, 
            right_boundary, lower_boundary, upper_boundary], origin='lower', 
                                        cmap='RdPu', alpha=0.5)
        
        self.circles = [cell.draw(ax1) for cell in self.cells]
        
        num_frames = int(total_time / frames_per_second)
        flux_matrix = np.zeros((num_frames, 100))
    
        # Add the given frame to the animation
        def animate(frame):
            print(f"Animating frame: {frame}")
            
            # Calculate the number of cell boundary crossings (for flux)
            time_step_crossings=self.advance_simulation(total_time/num_frames, 
                                                        diffusion_coeff=D)
            x_positions = [cell.x for cell in self.cells]
            
            # Get the density of cells. Store density data
            _, density = self.calculate_kde(x_positions, len(self.cells))
            self.density_matrix[frame, :] = density
            
            # Store flux data
            for i, boundary in enumerate(sorted(self.boundaries)):
                flux_matrix[frame, i] = time_step_crossings.get(boundary, 0) /\
                    (total_simulation_time/frames_per_second)

            # Plot concentration gradient
            concentration_plot.set_data(self.c)
            
            for i, cell in enumerate(self.cells):
                self.circles[i].center = cell.r
    
            ax1.set_title(f"Time: {frame * total_time / num_frames:.1f}s", 
                          fontsize=14)
            
            # Update cell positions and chemokine concentration profile
            if threeDPlot:
                ax2.clear()
                ax2.set_xlim(left_boundary, right_boundary)
                ax2.set_ylim(lower_boundary, upper_boundary)
                ax2.set_zlim(0, 6.5)
                ax2.set_xlabel("Position - x $(\mu m)$")
                ax2.set_ylabel("Position - y $(\mu m)$")
                ax2.set_xticks(np.arange(left_boundary,
                                         right_boundary + 1, 300))
                ax2.set_yticks(np.arange(lower_boundary, 
                                         upper_boundary + 1, 400))
                ax2.set_zlabel(r"$c^*$")
                ax2.set_title(r"$c^*(x,y,t)$")
                surf = ax2.plot_surface(self.X, self.Y, self.c, cmap='RdPu', 
                                        edgecolor='none')
                return self.circles + [concentration_plot, surf]
            else:
                return self.circles + [concentration_plot]
            
            plt.show()
            
        # Animate frames
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                       interval=50, blit=False)
        # Save animation
        if save:
            if threeDPlot:
                anim.save('hybrid_model_with_C.mp4', writer='ffmpeg', 
                          fps=30,dpi=100)
            else:
                anim.save('hybrid_model.mp4', writer='ffmpeg', fps=30, dpi=100)
            
        # Save density matrix 
        np.savetxt("density_matrix.txt", self.density_matrix, delimiter=",")
        self.plot_density_matrix()
    
        # Save flux matrix
        self.plot_flux_matrix(flux_matrix)
        

# Create and run the simulation
sim = Simulation()
sim.do_animation(total_time=total_simulation_time,
                 num_frames=total_simulation_time/frames_per_second, save=True)
sim.print_final_msd()