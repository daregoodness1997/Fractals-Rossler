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

# Print all values
for i in range(len(t_values)):
    print(f"t = {t_values[i]:.6f}, x = {y_values[i][0]:.6f}, y = {y_values[i][1]:.6f}, z = {y_values[i][2]:.6f}")

# # Create an instance of GrahamSmithOrthogonalization with the trajectory data 
# gso = GrahamSmithOrthogonalization(y_values)

# # Orthogonalize and normalize the vectors (each row is a vector in state space)
# orthogonal_vectors = gso.orthogonalize()
# orthonormal_vectors = gso.normalize()

# # Print orthonormal vectors for analysis (if needed)
# print("Orthonormal Vectors:")
# print(orthonormal_vectors)

# Create a figure with four subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

current_size=0.2

# Plot individual state variables
ax1.plot(t_values, y_values[:, 0], label='x(t)')
ax1.plot(t_values, y_values[:, 1], label='y(t)')
ax1.plot(t_values, y_values[:, 2], label='z(t)')
ax1.set_title(f"Rossler System Dynamics (a={a}, b={b}, c={c})")
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('State Variables')
ax1.legend()
ax1.grid(True)

# Plot phase space trajectory (x vs y)
ax2.plot(y_values[:, 0], y_values[:, 1])
ax2.set_title('Phase Space Trajectory (x vs y)')
ax2.set_xlabel('x(t)')
ax2.set_ylabel('y(t)')
ax2.grid(True)

# Set up the figure and axis for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Rössler System Dynamics in 3D')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')

# Initialize a scatter plot object
scat = ax.scatter([], [], [],s=2)

# Set limits for axes based on the data range (optional)
ax.set_xlim(np.min(y_values[:, 0]), np.max(y_values[:, 0]))
ax.set_ylim(np.min(y_values[:, 1]), np.max(y_values[:, 1]))
ax.set_zlim(np.min(y_values[:, 2]), np.max(y_values[:, 2]))


# Calculate number of frames and interval for an 8-second animation
num_frames = len(t_values)
duration_seconds = 8
interval_ms = (duration_seconds * 10) / num_frames

# Animation update function
def update(frame):
    scat._offsets3d = (y_values[:frame + 1, 0], y_values[:frame + 1, 1], y_values[:frame + 1, 2])
    return scat,

# Create the animation object
ani = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, blit=False)

plt.show()

