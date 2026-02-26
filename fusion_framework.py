import numpy as np
from ukf import BatteryUKF
from normalization import RobustAbsoluteDeviationNormalization
import sr as sr_module
import transfer as tl_module
from adaptivetransfer import AdaptiveScaleAdapter
from sparseadapter import SparseDataAdapter


class ComprehensiveBatteryPrognosisSystem:
    """
    Comprehensive Battery Prognosis System combining UKF, SR, and Robust Normalization
    """
    def __init__(self, mat_file_path):
        self.mat_file_path = mat_file_path
        self.data = None
        self.Max_Ic = None
        self.Max_Yc = None
        
        # Initialize components
        self.universal_model = None
        self.universal_scaler = None
        self.ukf_models = {}  # Dictionary to store UKF models for each pack
        self.normalizers = {}  # Dictionary to store normalization models
        self.transfer_adapters = {}  # Dictionary to store transfer adapters
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load data from mat file"""
        import scipy.io as sio
        print(f"--- Loading data from: {self.mat_file_path} ---")
        try:
            data = sio.loadmat(self.mat_file_path)
            self.Max_Ic = data['Max_Ic']
            self.Max_Yc = data['Max_Yc']
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Error: .mat file not found")
            raise
    
    def train_universal_sr_model(self, train_cell_indices):
        """
        Train universal SR model using 8 cell datasets (test1 to test8)
        """
        print("\n=== Training Universal SR Model (Cells 1-8) ===")
        
        # Prepare data from 8 training cells
        all_cycles = [self.Max_Ic[0, i].flatten() for i in train_cell_indices]
        all_caps = [self.Max_Yc[0, i].flatten() for i in train_cell_indices]
        
        # Train universal model using SR
        gp, scaler, score, formula = sr_module.train_universal_sr_model(all_cycles, all_caps)
        
        if gp:
            self.universal_model = gp
            self.universal_scaler = scaler
            print(f"  -> Training completed | R2 score: {score:.4f}")
            print(f"  -> Discovered formula: Rate = {formula}")
        else:
            print("  -> Training failed")
            raise Exception("SR training failed")
    
    def initialize_pack_prediction(self, pack_index, calibration_cycles=200):
        """
        Initialize prediction for a specific pack using first N cycles of data
        """
        print(f"\n=== Initializing Prediction for Pack {pack_index+1} ===")
        
        # Get the first calibration_cycles of data for this pack
        full_cycles = self.Max_Ic[0, pack_index].flatten()
        full_caps = self.Max_Yc[0, pack_index].flatten()
        
        # Use only the first calibration_cycles for initialization
        mask_calib = full_cycles <= calibration_cycles
        cycles_calib = full_cycles[mask_calib]
        caps_calib = full_caps[mask_calib]
        
        if len(caps_calib) == 0:
            raise ValueError(f"No calibration data available for pack {pack_index}")
        
        # Initialize UKF with initial capacity
        initial_capacity = caps_calib[0]  # First measured capacity
        degradation_rate_guess = (caps_calib[-1] - caps_calib[0]) / len(caps_calib)  # Rough estimate
        
        ukf_model = BatteryUKF(initial_capacity, degradation_rate_guess)
        
        # Initialize normalization for this pack
        normalizer = RobustAbsoluteDeviationNormalization()
        
        # Store the models
        self.ukf_models[pack_index] = ukf_model
        self.normalizers[pack_index] = normalizer
        
        print(f"  -> Pack {pack_index+1} initialized with {len(caps_calib)} calibration cycles")
        print(f"  -> Initial capacity: {initial_capacity:.4f}")
        
        return cycles_calib, caps_calib
    
    def compute_scaling_coefficient(self, pack_index, calibration_cycles=200):
        """
        Compute scaling coefficient using transfer learning between SR model and pack data
        """
        print(f"\n=== Computing Scaling Coefficient for Pack {pack_index+1} ===")
        
        # Get calibration data
        full_cycles = self.Max_Ic[0, pack_index].flatten()
        full_caps = self.Max_Yc[0, pack_index].flatten()
        
        mask_calib = full_cycles <= calibration_cycles
        cycles_calib = full_cycles[mask_calib]
        caps_calib = full_caps[mask_calib]
        
        # Initialize transfer adapter based on data characteristics
        if self.universal_model is None or self.universal_scaler is None:
            raise ValueError("Universal model not trained yet")
        
        # Choose adapter based on calibration data sparsity
        if len(cycles_calib) < 20:  # If calibration data is sparse
            adapter = SparseDataAdapter(self.universal_model, self.universal_scaler)
        else:
            adapter = AdaptiveScaleAdapter(self.universal_model, self.universal_scaler)
        
        # Compute scaling coefficient
        best_coe = adapter.fit_calibration_data(cycles_calib, caps_calib)
        
        # Store the adapter
        self.transfer_adapters[pack_index] = adapter
        
        print(f"  -> Optimized scaling coefficient (Coe): {best_coe:.4f}")
        if best_coe > 1:
            print("     (Interpretation: Target pack degrades slower than universal model)")
        else:
            print("     (Interpretation: Target pack degrades faster than universal model)")
        
        return best_coe
    
    def predict_with_ukf_sr_fusion(self, pack_index, future_cycles, initial_capacity, calibration_cycles=200):
        """
        Predict future capacity using improved fusion of UKF and SR methods
        """
        print(f"\n=== Predicting Future Capacity for Pack {pack_index+1} ===")
        
        # Get the transfer adapter
        if pack_index not in self.transfer_adapters:
            raise ValueError(f"No transfer adapter found for pack {pack_index}")
        
        adapter = self.transfer_adapters[pack_index]
        
        # Get the UKF model
        if pack_index not in self.ukf_models:
            raise ValueError(f"No UKF model found for pack {pack_index}")
        
        ukf_model = self.ukf_models[pack_index]
        
        # Use the transfer adapter to predict based on SR model
        pred_curve_sr = adapter.predict_trajectory(future_cycles, initial_capacity)
        
        # Run UKF prediction with adaptive fusion strategy
        pred_curve_ukf = []
        current_capacity = initial_capacity
        
        # Adaptive weighting based on prediction confidence
        sr_weight = 0.7  # Higher weight to SR as it's typically more reliable for long-term
        ukf_weight = 0.3  # Lower weight to UKF for stability
        
        for i, cycle in enumerate(future_cycles):
            # Predict next state using UKF
            dt = 1  # Assuming unit time step
            ukf_model.predict_degradation(dt)
            
            # Get UKF prediction
            ukf_state = ukf_model.get_current_state()
            ukf_capacity = ukf_state[0]
            
            # Get SR prediction for this step
            sr_prediction = pred_curve_sr[i]
            
            # Adaptive fusion: blend predictions based on confidence
            # Early in prediction, trust UKF more; later, trust SR more
            adaptive_factor = min(1.0, i / len(future_cycles) * 0.5 + 0.5)  # Gradually increase SR weight
            final_prediction = adaptive_factor * sr_prediction + (1 - adaptive_factor) * ukf_capacity
            
            pred_curve_ukf.append(final_prediction)
            
            # Update UKF with the final prediction to maintain consistency
            ukf_model.update_with_measurement(final_prediction)
        
        return np.array(pred_curve_sr), np.array(pred_curve_ukf)
    
    def calculate_failure_threshold(self, pack_index):
        """
        Calculate failure threshold as 80% of initial capacity
        """
        # Get the initial capacity (first value in the series)
        initial_capacity = self.Max_Yc[0, pack_index].flatten()[0]
        failure_threshold = 0.8 * initial_capacity
        
        return failure_threshold
    
    def evaluate_prediction_accuracy(self, pack_index, cycles_future, pred_curve, true_caps):
        """
        Evaluate prediction accuracy
        """
        if len(pred_curve) != len(true_caps):
            print(f"Warning: Prediction and true data lengths don't match for pack {pack_index+1}")
            min_len = min(len(pred_curve), len(true_caps))
            pred_curve = pred_curve[:min_len]
            true_caps = true_caps[:min_len]
        
        # Calculate various error metrics
        mse = np.mean((pred_curve - true_caps) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_curve - true_caps))
        mape = np.mean(np.abs((pred_curve - true_caps) / true_caps)) * 100
        
        # Calculate R² score
        ss_res = np.sum((true_caps - pred_curve) ** 2)
        ss_tot = np.sum((true_caps - np.mean(true_caps)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }