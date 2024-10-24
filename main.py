from fractal_rossler.rossler_system import RosslerSystem
from fractal_rossler.solver import RungeKutta4
from fractal_rossler.grahamsmith_orthogonalization import GrahamSmithOrthogonalization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initial conditions for (x, y, z)
initial_conditions = [1, 1, 1]
t0 = 0
tf = 1000
dt = 0.01
a = 0.2
b = 0.2
c = 5.7

# Create the Runge-Kutta solver instance
rossler = RosslerSystem.system
rk_solver = RungeKutta4(rossler, initial_conditions, t0, tf, dt)

# Solve the system
t_values, y_values = rk_solver.solve()

# Print all values (optional)
for i in range(len(t_values)):
    print(f"t = {t_values[i]:.6f}, x = {y_values[i][0]:.6f}, y = {y_values[i][1]:.6f}, z = {y_values[i][2]:.6f}")



# Set up the figure and axis for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Rössler System Dynamics in 3D') 
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')

# Initialize a scatter plot object with reduced size (e.g., s=5)
scat = ax.scatter([], [], [], s=5)  # Adjust 's' to change dot size

# Set limits for axes based on the data range (optional)
ax.set_xlim(np.min(y_values[:, 0]), np.max(y_values[:, 0]))
ax.set_ylim(np.min(y_values[:, 1]), np.max(y_values[:, 1]))
ax.set_zlim(np.min(y_values[:, 2]), np.max(y_values[:, 2]))

# # Calculate number of frames and interval for an 8-second animation
# num_frames = len(t_values)
# duration_seconds = 8
# interval_ms = (duration_seconds * 1000) / num_frames

# # Animation update function
# def update(frame):
#     scat._offsets3d = (y_values[:frame + 1, 0], y_values[:frame + 1, 1], y_values[:frame + 1, 2])
#     return scat,

# # Create the animation object
# ani = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False)

# def calculate_lyapunov_exponent(y_values, dt):
#     n = len(y_values)
#     # Initialize perturbations (small delta)
#     delta = np.eye(3) * 1e-6  # Small perturbation for each dimension
#     lyapunov_exponents = np.zeros(3)

#     # Store the logarithm of the lengths of perturbed trajectories
#     log_lengths = []

#     for i in range(1, n):
#         # Compute the new state with perturbation
#         y_current = y_values[i]
#         y_previous = y_values[i - 1]

#         # Calculate the Jacobian matrix (approximated)
#         jacobian = np.array([
#             [-y_previous[1], -y_previous[0], 0],
#             [y_previous[2], a - y_previous[0]],  # Corrected row
#             [b, y_previous[0]]                   # Corrected row
#         ])

#         # Update the perturbations using the Jacobian
#         delta = np.dot(jacobian, delta)

#         # Normalize to prevent numerical instability and keep track of lengths
#         norm_delta = np.linalg.norm(delta)
#         log_lengths.append(np.log(norm_delta))

#         # Re-normalize delta to avoid overflow/underflow issues
#         delta /= norm_delta

#     # Calculate average Lyapunov exponents from log lengths
#     lyapunov_exponents = np.mean(log_lengths) / dt

#     return lyapunov_exponents

# # Calculate Lyapunov Exponent for the Rössler system trajectory
# lyapunov_exponent = calculate_lyapunov_exponent(y_values, dt)

# Create 3D plot of z vs y vs x
fig3d = plt.figure(figsize=(10, 8))
ax3 = fig3d.add_subplot(111, projection='3d')
ax3.scatter(y_values[:, 0], y_values[:, 1], y_values[:, 2])
ax3.set_title('Rossler System Dynamics in 3D')
ax3.set_xlabel('x(t)')
ax3.set_ylabel('y(t)')
ax3.set_zlabel('z(t)')
plt.show()





