import numpy as np


class RobustAbsoluteDeviationNormalization:
    """
    Robust Absolute Deviation Normalization implementation
    This normalization method is robust to outliers compared to standard z-score normalization
    """
    def __init__(self):
        self.median = None
        self.mad = None  # Median Absolute Deviation
        self.fitted = False
        
    def fit(self, X):
        """
        Fit the normalization parameters to the data
        
        Args:
            X: Input data array (can be 1D or 2D)
        """
        X = np.asarray(X)
        
        # Calculate median along axis 0 (for 2D arrays) or axis 0 (for 1D arrays)
        if X.ndim == 1:
            self.median = np.median(X)
            # Calculate MAD (Median Absolute Deviation)
            self.mad = np.median(np.abs(X - self.median))
        else:
            self.median = np.median(X, axis=0)
            # Calculate MAD for each feature
            self.mad = np.median(np.abs(X - self.median), axis=0)
        
        # To avoid division by zero, set minimum MAD to a small value
        self.mad = np.maximum(self.mad, 1e-8)
        
        self.fitted = True
        
    def transform(self, X):
        """
        Transform the data using the fitted parameters
        
        Args:
            X: Input data array to normalize
            
        Returns:
            Normalized data array
        """
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet. Call fit() first.")
            
        X = np.asarray(X)
        
        # Apply robust normalization: (X - median) / MAD
        normalized_X = (X - self.median) / self.mad
        
        return normalized_X
        
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X: Input data array
            
        Returns:
            Normalized data array
        """
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X_normalized):
        """
        Inverse transformation to recover original scale
        
        Args:
            X_normalized: Normalized data array
            
        Returns:
            Original scale data array
        """
        if not self.fitted:
            raise ValueError("Normalizer has not been fitted yet. Call fit() first.")
            
        X_original = X_normalized * self.mad + self.median
        
        return X_original


def robust_normalize_with_bounds(data, lower_percentile=5, upper_percentile=95):
    """
    Additional robust normalization function with percentile-based bounds
    
    Args:
        data: Input data array
        lower_percentile: Lower percentile for clipping
        upper_percentile: Upper percentile for clipping
        
    Returns:
        Normalized data array
    """
    data = np.asarray(data)
    
    # Calculate percentiles
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    # Clip data to bounds
    clipped_data = np.clip(data, lower_bound, upper_bound)
    
    # Apply robust normalization
    normalizer = RobustAbsoluteDeviationNormalization()
    normalized_data = normalizer.fit_transform(clipped_data)
    
    return normalized_data, normalizer


def adaptive_robust_normalization(data, window_size=10):
    """
    Adaptive robust normalization that considers local statistics
    
    Args:
        data: Input data array
        window_size: Size of the sliding window for local statistics
        
    Returns:
        Normalized data array
    """
    data = np.asarray(data)
    n_points = len(data)
    normalized_data = np.zeros_like(data)
    
    for i in range(n_points):
        # Define the window around the current point
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_points, i + window_size // 2 + 1)
        
        # Extract local data
        local_data = data[start_idx:end_idx]
        
        # Calculate local median and MAD
        local_median = np.median(local_data)
        local_mad = np.median(np.abs(local_data - local_median))
        local_mad = max(local_mad, 1e-8)  # Avoid division by zero
        
        # Normalize the current point using local statistics
        normalized_data[i] = (data[i] - local_median) / local_mad
        
    return normalized_data