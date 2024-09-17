from fractal_rossler.rossler_system import RosslerSystem
from fractal_rossler.solver import RungeKutta4
from fractal_rossler.grahamsmith_orthogonalization import GrahamsmithOrthogonalization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initial conditions for (x, y, z)
initial_conditions = [1, 1, 1]
t0 = 0
tf = 1000
dt = 0.001
a = 0.2
b = 0.2
c = 5.7
t_max = 100


# Create the Runge-Kutta solver instance
rossler = RosslerSystem.system
rk_solver = RungeKutta4(rossler, initial_conditions, t0, tf, dt)
# # Create an instance of GrahamsmithOrthogonalization
# gso = GrahamsmithOrthogonalization(rossler,initial_conditions, a, b, c, t_max, dt)


# Solve the system
t_values, y_values = rk_solver.solve()

# Print all values
for i in range(len(t_values)):
    print(f"t = {t_values[i]:.6f}, x = {y_values[i][0]:.6f}, y = {y_values[i][1]:.6f}, z = {y_values[i][2]:.6f}")




# # Now call the method on the instance
# lyapunov_exponents, lyapunov_dimension = gso.grahamsmith_orthogonalization()

# # Print results
# print("\nLyapunov Exponents:", lyapunov_exponents)
# print("Lyapunov Dimension:", lyapunov_dimension)

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

# Create 3D plot of z vs y vs x
fig3d = plt.figure(figsize=(10, 8))
ax3 = fig3d.add_subplot(111, projection='3d')
ax3.scatter(y_values[:, 0], y_values[:, 1], y_values[:, 2])
ax3.set_title('Rossler System Dynamics in 3D')
ax3.set_xlabel('x(t)')
ax3.set_ylabel('y(t)')
ax3.set_zlabel('z(t)')
plt.show()