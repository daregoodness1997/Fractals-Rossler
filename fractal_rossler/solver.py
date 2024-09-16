import numpy as np

class RungeKutta4:
    def __init__(self, func, initial_conditions, t0, tf, dt):
        """
        Initialize the Runge-Kutta solver.

        Parameters:
        - func: The system of ODEs to be solved. Should be a function of (t, y).
        - initial_conditions: Array of initial values for the dependent variables.
        - t0: Initial time.
        - tf: Final time.
        - dt: Time step size.
        """
        self.func = func
        self.initial_conditions = np.array(initial_conditions)
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.n_steps = int((tf - t0) / dt)
        self.num_variables = len(initial_conditions)

    def solve(self):
        """
        Solve the system of ODEs using the fourth-order Runge-Kutta method.

        Returns:
        - t_values: Array of time points where the solution is computed.
        - y_values: Array of the solution for each dependent variable.
        """
        t_values = np.linspace(self.t0, self.tf, self.n_steps + 1)
        y_values = np.zeros((self.n_steps + 1, self.num_variables))
        y_values[0] = self.initial_conditions
        
        for i in range(self.n_steps):
            t = t_values[i]
            y = y_values[i]
            
            # Compute Runge-Kutta coefficients
            k1 = self.dt * self.func(t, y)
            k2 = self.dt * self.func(t + self.dt / 2, y + k1 / 2)
            k3 = self.dt * self.func(t + self.dt / 2, y + k2 / 2)
            k4 = self.dt * self.func(t + self.dt, y + k3)
            
            # Update the next value of y
            y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        return t_values, y_values




