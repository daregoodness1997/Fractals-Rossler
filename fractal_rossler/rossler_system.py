import numpy as np


class RosslerSystem:
    def __init__(self,t,y):
        """
    Define the Rossler system of ODEs.

    Parameters:
    - t: Time (independent variable).
    - y: Array of dependent variables (x, y, z).

    Returns:
    - dydt: Array of derivatives [dx/dt, dy/dt, dz/dt].
    """
        self.t = t
        self.y = y
    
    def system(t,y):
        a, b, c = 0.2, 0.2, 5.7  # Parameters for the Rossler system
        x, y, z = y
        dxdt = -y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
    
        return np.array([dxdt, dydt, dzdt])

    
