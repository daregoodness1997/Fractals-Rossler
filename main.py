from fractal_rossler.rossler_system import RosslerSystem
from fractal_rossler.solver import RungeKutta4
from fractal_rossler.grahamsmith_orthogonalization import GrahamsmithOrthogonalization

# Initial conditions for (x, y, z)
initial_conditions = [1, 1, 1]
t0 = 0
tf = 100
dt = 0.001
a = 0.2
b = 0.2
c = 5.7
t_max = 100

# Create an instance of GrahamsmithOrthogonalization
gso = GrahamsmithOrthogonalization(initial_conditions, a, b, c, t_max, dt)

# Create the Runge-Kutta solver instance
rossler = RosslerSystem.system
rk_solver = RungeKutta4(rossler, initial_conditions, t0, tf, dt)

# Solve the system
t_values, y_values = rk_solver.solve()

# Print all values
for i in range(len(t_values)):
    print(f"t = {t_values[i]:.4f}, x = {y_values[i][0]:.4f}, y = {y_values[i][1]:.4f}, z = {y_values[i][2]:.4f}")

# Now call the method on the instance
lyapunov_exponents, lyapunov_dimension = gso.grahamsmith_orthogonalization()

# Print results
print("\nLyapunov Exponents:", lyapunov_exponents)
print("Lyapunov Dimension:", lyapunov_dimension)