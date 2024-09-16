from fractal_rossler.solver import RungeKutta4
import numpy as np

class GrahamsmithOrthogonalization:
    def __init__(self, initial_conditions, a, b, c, t_max, dt):
        self.initial_conditions = initial_conditions
        self.a = a
        self.b = b
        self.c = c
        self.t_max = t_max
        self.dt = dt

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
            def system(t, y):
                return [self.a * (y[1] - y[2]),
                        self.b * (y[2] + y[0]),
                        self.c * (y[1] - y[0])]

            # RK4 integration
            rk4_solver = RungeKutta4(system, self.initial_conditions, 0, self.t_max, self.dt)
            states[i] = rk4_solver.solve()[1][i]

            # Update orthogonalization matrices
            A = np.dot(A, states[i] - states[i-1])
            Q, R = np.linalg.qr(A)  # QR decomposition

        # Compute Lyapunov exponents
        lyapunov_exponents = np.log(np.abs(np.diag(R)))

        # Estimate Lyapunov dimension
        lyapunov_dimension = np.sum(lyapunov_exponents > 0)

        return lyapunov_exponents, lyapunov_dimension