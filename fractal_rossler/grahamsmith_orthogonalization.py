from fractal_rossler.solver import RungeKutta4
from fractal_rossler.rossler_system import RosslerSystem
import numpy as np



class GrahamsmithOrthogonalization:
    def __init__(self,func, initial_conditions, a, b, c, t_max, dt):
        self.initial_conditions = initial_conditions
        self.a = a
        self.b = b
        self.c = c
        self.t_max = t_max
        self.dt = dt
        self.func = func


    def grahamsmith_orthogonalization(self):
        num_vars = len(self.initial_conditions)
        num_steps = int(self.t_max / self.dt)
        states = np.zeros((num_steps, num_vars))
        states[0] = self.initial_conditions
        t_values = np.linspace(0, self.t_max, num_steps)
        
        # Setup identity matrices
        A = np.eye(num_vars)
        Q = np.eye(num_vars)
        R = np.zeros((num_vars, num_vars))

        # Time integration using RK4 solver
        for i in range(1, num_steps):
            # Define the system function (assuming it's defined elsewhere)
            rossler = RosslerSystem.system


            # RK4 integration
            rk4_solver = RungeKutta4(rossler, self.initial_conditions, 0, self.t_max, self.dt)
            states[i] = rk4_solver.solve()[1][i]

            # Accumulate state differences
            diff_matrix = np.zeros((num_steps - 1, num_vars))
            for i in range(1, num_steps):
                diff_matrix[i-1] = states[i] - states[i-1]

            # Perform QR decomposition on the accumulated differences
            Q, R = np.linalg.qr(diff_matrix.T)

            # Compute Lyapunov exponents
            lyapunov_exponents = np.log(np.abs(np.diag(R)))
            
            # Check for division by zero
            safe_divisor = self.dt * (num_steps - 1) + 1e-10  # Add a small epsilon
            
            lyapunov_exponents /= safe_divisor
            
            # Estimate Lyapunov dimension
            lyapunov_dimension = np.sum(lyapunov_exponents > 0)
            
        return lyapunov_exponents, lyapunov_dimension