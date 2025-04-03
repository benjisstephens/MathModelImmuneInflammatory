import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import pandas as pd
from scipy.signal.windows import gaussian
from scipy.stats import truncnorm
import random
import seaborn as sns 

def RunHybridModel(D, chi):

    plt.rcParams.update({'font.size': 14})

    threeDPlot = False  # True: display both 2D and 3D views;
                        # False: display only 2D;

    chemokine_present = True
    flow_direction = 'NEG'

    if chemokine_present == False:
        data_file = "M4_wDC_CTRL_POS_pos_export.txt"
        U = 2 * 8.4e-2          # um s^-1
        proportion_phi_c = 0
    else:
        if flow_direction == 'NEG':
            U = -2 * 8.4e-2  # um s^-1
            data_file = "M12_wDC_CCL21_NEG_pos_export.txt"
            proportion_phi_c = 0.2
        elif flow_direction == 'DIF':
            data_file = "M12_wDC_CCL21_DIF_pos_export.txt"
            U = 0  # um s^-1
            proportion_phi_c = 0.36
        elif flow_direction == 'POS':
            data_file = "M12_wDC_CCL21_POS_pos_export.txt"
            U = 2 * 8.4e-2  # um s^-1
            proportion_phi_c = 0.36

    # %% Boundaries
    left_boundary = 0
    right_boundary = 1200
    lower_boundary = 0
    upper_boundary = 1600

    # %% Animation parameters
    total_simulation_time = 1800
    frames_per_second = 18

    # %% Stochastic difference equation parameters
    k_N7_p = 1.83e-3    # Rate phi -> phi_c in s^-1 molecules^-1 um^3
    k_N7_m = 5e-3       # Rate phi_c -> phi in s^-1

    # Define gaussian function to edit chemokine concentration when
    #   binding/unbinding occurs
    def change_chemokine_gradient(x, y, x0, y0):
        # return 1e-3 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * 100**2))
        return 0.01 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * 100**2))

    # %% Class which defines a cell
    class Cell:
        def __init__(self, x, y, radius, is_bound=False):
            # Position of cell
            self.r = np.array((x, y))
            # Radius of cell
            self.radius = radius
            # Flag to differentiate between phi and phi_c
            self.is_bound = is_bound

            # Style cells appropriately
            if self.is_bound:
                self.circle = Circle(xy=self.r, radius=self.radius,
                                     edgecolor='red', facecolor='red', fill=True)
            else:
                self.circle = Circle(xy=self.r, radius=self.radius,
                                     edgecolor='green', facecolor='green', fill=True)

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

        def check_for_overlap(self, other):
            return np.hypot(*(self.r - other.r)) < self.radius + other.radius

        def draw(self, ax):
            ax.add_patch(self.circle)
            return self.circle

        def advance_cell(self, dt, diffusion_coeff, concentration, gradient_func):
            # Cells move by diffusion
            proposed_r = self.r + \
                np.sqrt(2*diffusion_coeff*dt)*np.random.randn(2)

            # Bound cells move by chemotaxis
            if self.is_bound:
                proposed_r += chi * \
                    gradient_func(self.x, self.y, concentration)*dt

            # Cells cannot move outside of boundaries
            if left_boundary + self.radius <= proposed_r[0] <= right_boundary \
                - self.radius and \
               lower_boundary + self.radius <= proposed_r[1] <= upper_boundary \
                    - self.radius:
                self.r = proposed_r
            else:
                self.r = proposed_r  # If out of bounds the cell simply remains

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

        def modify_concentration(self, concentration, is_binding):
            x, y = self.x, self.y
            x_indices = np.linspace(left_boundary, right_boundary,
                                    concentration.shape[1])
            y_indices = np.linspace(lower_boundary, upper_boundary,
                                    concentration.shape[0])
            X, Y = np.meshgrid(x_indices, y_indices)
            gaussian = change_chemokine_gradient(X, Y, x, y)
            if is_binding:
                concentration -= gaussian
            else:
                concentration += gaussian
            concentration = np.clip(concentration, 0, None)

    # %% Class which defines a simulation
    class Simulation:
        def __init__(self, radius=6):
            self.initial_positions = {}
            self.init_cells(radius)
            self.initialise_concentration()
            self.boundaries = np.linspace(
                left_boundary, right_boundary, 100)[1:-1]
            self.boundary_crossings = {b: 0 for b in self.boundaries}
            self.num_frames = int(total_simulation_time / frames_per_second)
            self.density_matrix = np.zeros((self.num_frames, 100))
            self.final_positions = {}

        def init_cells(self, radius):
            self.cells = []
            # Load data from file
            df = pd.read_csv(data_file)

            # Filter for cells at Time(s) = 2.8
            df_filtered = df[df['Time(s)'].round(1) == 2.8]
            # Shuffle and split into two equal groups for is_bound assignment
            df_shuffled = df_filtered.sample(frac=1,
                                             random_state=42).reset_index(drop=True)
            proportion_bound = len(df_shuffled) * (proportion_phi_c)

            for i, row in df_shuffled.iterrows():
                x, y = row['x(microns)'], row['y(microns)']
                transformed_x, transformed_y = 1250 - y, x
                is_bound = i < proportion_bound
                cell = Cell(transformed_x, transformed_y,
                            radius, is_bound=is_bound)
                self.cells.append(cell)
                for cell in self.cells:
                    self.initial_positions[cell] = np.array(cell.r)

        def advance_simulation(self, dt, diffusion_coeff):
            previous_positions = {cell: cell.x for cell in self.cells}

            for cell in self.cells:
                cell.advance_cell(dt, diffusion_coeff, self.c,
                                  self.gradient_concentration)
                local_concentration = self.get_local_concentration(
                    cell.x, cell.y)
                cell.update_type(k_N7_p*local_concentration *
                                 dt, k_N7_m*dt, self.c)

            for cell in self.cells:
                self.final_positions[cell] = np.array(cell.r)

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

            return time_step_crossings

        def initialise_concentration(self):
            x = np.linspace(left_boundary, right_boundary, 100)
            y = np.linspace(lower_boundary, upper_boundary, 100)
            D_val = 100         # um^2 s^-1

            if chemokine_present == True:
                C = 6.022           # molecules um^-1
            else:
                C = 0
            d = 1.3e-6          # s^-1
            M1 = (U + np.sqrt(U**2 + 4 * D_val * d)) / (2 * D_val)
            M2 = (U - np.sqrt(U**2 + 4 * D_val * d)) / (2 * D_val)
            A = C / (1 - np.exp((M1 - M2) * right_boundary))
            self.X, self.Y = np.meshgrid(x, y)
            self.c = A * np.exp(M1 * self.X) + (C - A) * np.exp(M2 * self.X)
            self.c[:, 0] = C
            self.c[:, -1] = 0
            self.c = gaussian_filter(self.c, sigma=1)

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

        def get_local_concentration(self, x, y):
            dx = (self.c.shape[1] - 1) / (right_boundary - left_boundary)
            dy = (self.c.shape[0] - 1) / (upper_boundary - lower_boundary)
            ix = int(np.clip((x - left_boundary) * dx, 0, self.c.shape[1] - 1))
            iy = int(np.clip((y - lower_boundary)
                     * dy, 0, self.c.shape[0] - 1))
            return self.c[iy, ix]

        def calculate_kde(self, x_positions, cell_count):
            # Set bandwidth scaling using the number of cells (optimal bandwidth ~ n^(-1/5))
            kde = gaussian_kde(x_positions, bw_method=1 * cell_count**(-1/5))

            # Evaluate the KDE on the original domain [0, 1200]
            x_grid = np.linspace(0, 1200, 100)

            # Multiply by first_time_cells to scale the density to "cells per micron"
            density = kde(x_grid) * cell_count

            return x_grid, density

        def plot_flux_matrix(self, flux_matrix):
            flux_matrix_smoothed = gaussian_filter(
                flux_matrix.astype(float), sigma=5)
            size = int(6 * 5)
            gaussian_kernel = gaussian(size, 5)
            kernel_sum = np.sum(gaussian_kernel)
            flux_matrix_smoothed_normalized = flux_matrix_smoothed / kernel_sum

            return flux_matrix_smoothed_normalized

        def do_animation(self, total_time, num_frames, save=True):
            if threeDPlot:
                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(1, 2)

                # 2D subplot (left)
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.set_aspect('equal', 'box')
                ax1.set_xlim(left_boundary, right_boundary)
                ax1.set_ylim(lower_boundary, upper_boundary)
                ax1.set_xlabel("x (microns)")
                ax1.set_ylabel("y (microns)")
                ax1.set_xticks(
                    np.arange(left_boundary, right_boundary + 1, 200))
                ax1.set_yticks(
                    np.arange(lower_boundary, upper_boundary + 1, 200))
                ax1.tick_params(axis='both', which='major')

                # 3D subplot (right)
                ax2 = fig.add_subplot(gs[0, 1], projection='3d')
                ax2.set_xlim(left_boundary, right_boundary)
                ax2.set_ylim(lower_boundary, upper_boundary)
                ax2.set_xticks(
                    np.arange(left_boundary, right_boundary + 1, 400))
                ax2.set_yticks(
                    np.arange(lower_boundary, upper_boundary + 1, 400))
                ax2.set_zlim(0, 6.5)
                ax2.set_xlabel("x (microns)")
                ax2.set_ylabel("y (microns)")
                ax2.set_zlabel("Concentration C")
                ax2.set_title("3D View of Concentration C")

                plt.show()
            else:
                # Create only one figure and axis (2D view)
                fig, ax1 = plt.subplots(figsize=(8, 8))
                ax1.set_aspect('equal', 'box')
                ax1.set_xlim(left_boundary, right_boundary)
                ax1.set_ylim(lower_boundary, upper_boundary)
                ax1.set_xlabel("x (microns)")
                ax1.set_ylabel("y (microns)")
                ax1.set_xticks(
                    np.arange(left_boundary, right_boundary + 1, 200))
                ax1.set_yticks(
                    np.arange(lower_boundary, upper_boundary + 1, 200))

            # Plot the initial 2D concentration as an image (heatmap)
            concentration_plot = ax1.imshow(self.c, extent=[left_boundary, 
                            right_boundary, lower_boundary, upper_boundary],
                                    origin='lower', cmap='RdPu', alpha=0.5)

            # Draw the cells on the 2D axis
            self.circles = [cell.draw(ax1) for cell in self.cells]

            # Initialize flux_matrix and density_matrix (for saving later)
            num_frames = int(total_time / frames_per_second)
            flux_matrix = np.zeros((num_frames, 100))

            def animate(frame):

                # Advance the simulation
                time_step_crossings = self.advance_simulation(
                    total_time / num_frames, diffusion_coeff=D)

                # Update density matrix (using KDE on x positions)
                x_positions = [cell.x for cell in self.cells]
                _, density = self.calculate_kde(x_positions, len(self.cells))
                self.density_matrix[frame, :] = density

                # Update flux_matrix
                for i, boundary in enumerate(sorted(self.boundaries)):
                    flux_matrix[frame, i] = time_step_crossings.get(
                        boundary, 0) / (total_simulation_time/frames_per_second)

                # Update the 2D concentration image
                concentration_plot.set_data(self.c)

                # Update the location of the cells in the 2D plot
                for i, cell in enumerate(self.cells):
                    self.circles[i].center = cell.r

                ax1.set_title(
                    f"Time: {frame * total_time / num_frames:.1f}s", fontsize=14)

                # If threeDPlot is True, update the 3D subplot as well.
                if threeDPlot:
                    ax2.clear()
                    ax2.set_xlim(left_boundary, right_boundary)
                    ax2.set_ylim(lower_boundary, upper_boundary)
                    ax2.set_zlim(0, 6.5)
                    ax2.set_xlabel("Position - x $(\mu m)$")
                    ax2.set_ylabel("Position - y $(\mu m)$")
                    ax2.set_xticks(
                        np.arange(left_boundary, right_boundary + 1, 300))
                    ax2.set_yticks(
                        np.arange(lower_boundary, upper_boundary + 1, 400))
                    ax2.set_zlabel(r"$c^*$")
                    ax2.set_title(r"$c^*(x,y,t)$")
                    surf = ax2.plot_surface(
                        self.X, self.Y, self.c, cmap='RdPu', edgecolor='none')
                    return self.circles + [concentration_plot, surf]
                else:
                    return self.circles + [concentration_plot]

                plt.show()

            anim = animation.FuncAnimation(
                fig, animate, frames=num_frames, interval=50, blit=False)
            if save:
                if threeDPlot:
                    anim.save('hybrid_model_with_C.mp4',
                              writer='ffmpeg', fps=30, dpi=100)
                else:
                    anim.save('hybrid_model.mp4',
                              writer='ffmpeg', fps=30, dpi=100)


            # Save flux_matrix to a file and also plot it
            new_flux_matrix = self.plot_flux_matrix(flux_matrix)

            return self.density_matrix, new_flux_matrix

    # Create and run the simulation
    sim = Simulation()
    densityMatrix, fluxMatrix = sim.do_animation(
        total_time=total_simulation_time, 
        num_frames=total_simulation_time/frames_per_second,
        save=True)

    return densityMatrix, fluxMatrix

# Function to load the data

def load_data(filename):
    # Assuming data is stored in a text format
    return np.loadtxt(filename, delimiter=",")

# Function to compute the log-likelihood
def log_likelihood(F, y, sigma=100):
    diff = F - y[:, :-1]
    return (-1/2*sigma**2)*np.sum(diff** 2) - (np.prod(y.shape)/2)*np.log(2*np.pi*sigma**2)

def log_prior(D, chi):
    if D <= 0 or chi <= 0:  # Ensure parameters remain positive
        return -np.inf
    
    log_P_D = stats.norm(loc=1, scale=1).logpdf(D)  # Regular normal prior for D
    log_P_chi = stats.norm(loc=6, scale=1).logpdf(chi)  # Regular normal prior for Chi

    return log_P_D + log_P_chi


# Symmetric proposal distribution
def propose_theta(theta, step_size):
    return np.random.normal(theta, step_size)

def metropolis_hastings(data_filename, num_samples, step_size):
    # Load experimental data
    y = load_data(data_filename)

    # Initial parameter values
    D, chi = 2, 10
    theta = np.array([D, chi])

    # Run model with initial parameters
    densityMatrix_current, fluxMatrix_current = RunHybridModel(theta[0], theta[1])
    #log_like_current = log_likelihood(densityMatrix_current, y)
    log_like_current = log_likelihood(fluxMatrix_current, y)
    log_prior_current = log_prior(theta[0], theta[1])
    log_posterior_current = log_like_current# + log_prior_current
    
    
    # Store only accepted samples
    accepted_samples = [theta]

    for i in range(num_samples):
        print(f"\nIteration: {i}\n")

        # Propose new parameters
        theta_star = propose_theta(theta, step_size)
        D_star, chi_star = theta_star
        print(f"Proposed parameters: {theta_star}")

        # Ensure proposed values are within valid ranges
        if D_star <= 0 or chi_star <= 0:
            print("Rejected due to non-positive parameters.")
            continue  # Skip storing and plotting rejected parameters

        # Run model with proposed parameters
        print("Running simulation for proposed parameters...")
        densityMatrix_star, fluxMatrix_star = RunHybridModel(D_star, chi_star)
        
        
        # Compute log-likelihood and log-prior for the proposed state
        #log_like_star = log_likelihood(densityMatrix_star, y)
        log_like_star = log_likelihood(fluxMatrix_star, y)
        log_prior_star = log_prior(D_star, chi_star)
        log_posterior_star = log_like_star# + log_prior_star

        # Compute acceptance probability (in log-space)
        log_r = log_posterior_star - log_posterior_current
        print(f"Acceptance ratio (exp(log_r)): {np.exp(log_r)}")

        # Accept or reject the proposal
        random_var = random.uniform(0, 1)
        log_random_var = np.log(random_var)
        print(f"random variable for comparison: {random_var}")

        if log_random_var < log_r:
            theta = theta_star  # Accept move
            log_posterior_current = log_posterior_star
            log_like_current = log_like_star
            densityMatrix_current = densityMatrix_star
            fluxMatrix_current = fluxMatrix_star
            print("Proposal accepted.")
        else:
            print("Proposal rejected.")
            
        accepted_samples.append(theta)  # Store only accepted values
        
        np.savetxt("parameter_list.txt", accepted_samples, delimiter=",")
        
        # Plot only accepted samples
        if accepted_samples:
            accepted_samples_np = np.array(accepted_samples)  
            plt.figure(figsize=(6, 4))

            # Plot D
            plt.subplot(2, 1, 1)
            plt.plot(range(len(accepted_samples_np)), accepted_samples_np[:, 0],
                     linestyle='-', color='darkred')
            plt.ylabel(r'$D^{*}$')
            plt.grid()
            
            # Plot chi
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accepted_samples_np)), accepted_samples_np[:, 1],
                     linestyle='-', color='darkred')
            plt.xlabel('Iteration')
            plt.ylabel(r'$\chi^{*}$')
            plt.grid()

            plt.tight_layout()
            plt.show()

    return np.array(accepted_samples)

# Example function call
parameters = metropolis_hastings("M12_wDC_CCL21_NEG_flux_data.txt", 5000, (3, 6))

#parameters = np.loadtxt("parameter_list.txt", delimiter=",")

plt.figure(figsize=(6, 4))

# Plot D timeseries
plt.subplot(2, 1, 1)
plt.plot(range(len(parameters)), parameters[:, 0], linestyle='-', color='darkred')
plt.ylabel(r'$D^{*}$')
plt.grid()

# Plot chi timeseries
plt.subplot(2, 1, 2)
plt.plot(range(len(parameters)), parameters[:, 1], linestyle='-', color='darkred')
plt.xlabel('Iteration')
plt.ylabel(r'$\chi^{*}$')
plt.grid()

plt.tight_layout()
plt.show()

plt.close()

# Remove the first 10% of values (burn-in period)
parameters = parameters[int(len(parameters)/10):]

# Define the log-normal priors
num_prior_samples = 5000  # Number of prior samples
D_prior_samples = np.random.normal(loc=1, scale=1, size=num_prior_samples)
chi_prior_samples = np.random.normal(loc=6, scale=1, size=num_prior_samples)

# Transform back to original space
D_prior_samples = np.array(D_prior_samples)
chi_prior_samples = np.array(chi_prior_samples)

# Plot KDEs
plt.figure(figsize=(6, 5))

kde_D = gaussian_kde(parameters[:, 0], bw_method='scott')
D_values = np.linspace(0, 100, 1000)
D_density = kde_D(D_values)
D_modal_value = D_values[np.argmax(D_density)]  # Find the x-value at the peak

# KDE for chi (parameters[:, 1])
kde_chi = gaussian_kde(parameters[:, 1], bw_method='scott')
chi_values = np.linspace(0, 50, 1000)
chi_density = kde_chi(chi_values)
chi_modal_value = chi_values[np.argmax(chi_density)]  # Find the x-value at the peak

print(D_modal_value)
print(chi_modal_value)

# KDE for D
plt.subplot(2, 1, 1)
sns.kdeplot(parameters[:, 0], color='darkred', fill=True, label='Posterior')
#sns.kdeplot(D_prior_samples, color='darkred', linestyle='dashed', label='Prior')
plt.xlabel(r'$D^{*}$')
plt.xlim(0,20)
plt.grid()

# KDE for chi
plt.subplot(2, 1, 2)
sns.kdeplot(parameters[:, 1], color='darkred', fill=True, label='Posterior')
#sns.kdeplot(chi_prior_samples, color='darkred', linestyle='dashed', label='Prior')
plt.xlabel(r'$\chi^{*}$')
plt.xlim(0,20)
plt.grid()

plt.tight_layout()
plt.show()
