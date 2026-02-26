import numpy as np
from scipy.linalg import cholesky, qr


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter implementation for battery capacity prediction
    """
    def __init__(self, n_states, n_measurements, alpha=1e-3, beta=2, kappa=0):
        """
        Initialize UKF parameters
        
        Args:
            n_states: Number of state variables
            n_measurements: Number of measurement variables
            alpha: Spread of sigma points (typically 1e-4 to 1)
            beta: Prior knowledge of distribution (2 for Gaussian)
            kappa: Secondary scaling parameter (typically 0)
        """
        self.n_x = n_states  # State dimension
        self.n_z = n_measurements  # Measurement dimension
        
        # UKF scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Calculate composite scaling parameter
        lambda_ = alpha ** 2 * (n_states + kappa) - n_states
        self.lambda_ = lambda_
        
        # Weights for mean and covariance
        self.W_m = np.zeros(2 * n_states + 1)
        self.W_c = np.zeros(2 * n_states + 1)
        
        self.W_m[0] = lambda_ / (n_states + lambda_)
        self.W_c[0] = lambda_ / (n_states + lambda_) + (1 - alpha ** 2 + beta)
        
        for i in range(1, 2 * n_states + 1):
            self.W_m[i] = self.W_c[i] = 1.0 / (2 * (n_states + lambda_))
        
        # State and covariance
        self.x = np.zeros(n_states)  # State vector
        self.P = np.eye(n_states)    # Covariance matrix
        
        # Process and measurement noise
        self.Q = np.eye(n_states)    # Process noise
        self.R = np.eye(n_measurements)  # Measurement noise
        
    def initialize_state(self, x0, P0):
        """Initialize state and covariance"""
        self.x = np.copy(x0)
        self.P = np.copy(P0)
        
    def set_noise_covariances(self, Q, R):
        """Set process and measurement noise covariances"""
        self.Q = Q
        self.R = R
        
    def _sigma_points(self, x, P):
        """Generate sigma points"""
        n = len(x)
        lambda_plus_n = self.lambda_ + n
        U = cholesky(lambda_plus_n * P).T  # Use transpose to get upper triangular matrix
        
        sigma_points = np.zeros((n, 2 * n + 1))
        sigma_points[:, 0] = x
        
        for i in range(n):
            sigma_points[:, i + 1] = x + U[i, :]
            sigma_points[:, i + 1 + n] = x - U[i, :]
            
        return sigma_points
    
    def predict(self, fx_func, dt):
        """
        Prediction step of UKF
        
        Args:
            fx_func: State transition function
            dt: Time step
        """
        # Generate sigma points
        sigma_points = self._sigma_points(self.x, self.P)
        
        # Propagate sigma points through state transition function
        x_pred = np.zeros(self.n_x)
        P_pred = np.zeros((self.n_x, self.n_x))
        
        for i in range(2 * self.n_x + 1):
            sigma_point = sigma_points[:, i]
            propagated_point = fx_func(sigma_point, dt)
            x_pred += self.W_m[i] * propagated_point
            
        # Calculate predicted covariance
        for i in range(2 * self.n_x + 1):
            sigma_point = sigma_points[:, i]
            propagated_point = fx_func(sigma_point, dt)
            diff = propagated_point - x_pred
            P_pred += self.W_c[i] * np.outer(diff, diff)
            
        P_pred += self.Q
        
        # Update state and covariance
        self.x = x_pred
        self.P = P_pred
        
    def update(self, z, hx_func):
        """
        Update step of UKF
        
        Args:
            z: Measurement vector
            hx_func: Measurement function
        """
        # Generate sigma points from predicted state
        sigma_points = self._sigma_points(self.x, self.P)
        
        # Transform sigma points through measurement function
        z_pred = np.zeros(self.n_z)
        P_zz = np.zeros((self.n_z, self.n_z))
        P_xz = np.zeros((self.n_x, self.n_z))
        
        # Predict measurements
        for i in range(2 * self.n_x + 1):
            sigma_point = sigma_points[:, i]
            z_i = hx_func(sigma_point)
            z_pred += self.W_m[i] * z_i
            
        # Calculate innovation covariance and cross-covariance
        for i in range(2 * self.n_x + 1):
            sigma_point = sigma_points[:, i]
            z_i = hx_func(sigma_point)
            
            z_diff = z_i - z_pred
            x_diff = sigma_point - self.x
            
            P_zz += self.W_c[i] * np.outer(z_diff, z_diff)
            P_xz += self.W_c[i] * np.outer(x_diff, z_diff)
            
        P_zz += self.R
        
        # Calculate Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Update state and covariance
        innovation = z - z_pred
        self.x += K @ innovation
        self.P -= K @ P_zz @ K.T


class BatteryUKF:
    """
    Battery-specific UKF implementation for capacity prediction
    """
    def __init__(self, initial_capacity, degradation_rate_guess=0.001):
        """
        Initialize battery UKF
        
        Args:
            initial_capacity: Initial capacity of the battery
            degradation_rate_guess: Initial guess for degradation rate
        """
        # State: [capacity, degradation_rate]
        self.ukf = UnscentedKalmanFilter(n_states=2, n_measurements=1)
        
        # Initialize state: [capacity, degradation_rate]
        x0 = np.array([initial_capacity, degradation_rate_guess])
        # Initialize covariance
        P0 = np.diag([0.01, 0.0001])  # Small uncertainty in initial values
        
        self.ukf.initialize_state(x0, P0)
        
        # Set process and measurement noise
        Q = np.diag([1e-6, 1e-8])  # Process noise
        R = np.array([[0.001]])     # Measurement noise (variance of capacity measurements)
        self.ukf.set_noise_covariances(Q, R)
        
    def predict_degradation(self, dt):
        """
        State transition function for battery degradation
        State: [capacity, degradation_rate]
        """
        def fx(state, dt):
            capacity, degradation_rate = state
            # Simple linear degradation model: C(k+1) = C(k) + degradation_rate * dt
            new_capacity = capacity + degradation_rate * dt
            new_degradation_rate = degradation_rate  # Assume constant degradation rate
            return np.array([new_capacity, new_degradation_rate])
        
        self.ukf.predict(fx, dt)
        
    def update_with_measurement(self, measured_capacity):
        """
        Measurement update with capacity measurement
        """
        def hx(state):
            capacity, degradation_rate = state
            # Measurement is just the capacity
            return np.array([capacity])
        
        self.ukf.update(np.array([measured_capacity]), hx)
        
    def get_current_state(self):
        """Get current estimated state [capacity, degradation_rate]"""
        return self.ukf.x
        
    def get_state_covariance(self):
        """Get current state covariance"""
        return self.ukf.P