import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from fusion_framework import ComprehensiveBatteryPrognosisSystem
from draw import (plot_capacity_degradation_trajectory, plot_error_analysis, 
                  plot_scaling_coefficient, plot_model_performance_metrics, 
                  plot_rul_prediction, plot_prediction_uncertainty_analysis)
import os


def main():
    """
    Main function to execute the comprehensive battery prognosis system
    """
    # Configuration
    MAT_FILE_PATH = 'best_estimation.mat'
    TRAIN_CELL_INDICES = list(range(8))  # Cells 1-8 for training
    TARGET_PACK_INDICES = list(range(8, 14))  # Packs 9-14 for testing (0-indexed: 8-13)
    CALIBRATION_CYCLES = 200  # Use first 200 cycles for calibration
    FAILURE_THRESHOLD_RATIO = 0.8  # 80% of initial capacity as failure threshold
    
    print("=== Comprehensive Battery Prognosis System ===")
    print("Combining UKF, SR, and Robust Absolute Deviation Normalization")
    
    # Initialize the comprehensive system
    system = ComprehensiveBatteryPrognosisSystem(MAT_FILE_PATH)
    
    # Step 1: Train universal SR model using 8 cell datasets
    system.train_universal_sr_model(TRAIN_CELL_INDICES)
    
    # Step 2: Process each target pack (test9 to test14)
    results = {}
    
    for pack_idx in TARGET_PACK_INDICES:
        print(f"\n{'='*60}")
        print(f"Processing Pack {pack_idx + 1}")
        print(f"{'='*60}")
        
        # Initialize prediction for this pack
        cycles_calib, caps_calib = system.initialize_pack_prediction(pack_idx, CALIBRATION_CYCLES)
        
        # Compute scaling coefficient using transfer learning
        scaling_coe = system.compute_scaling_coefficient(pack_idx, CALIBRATION_CYCLES)
        
        # Get full data for evaluation
        full_cycles = system.Max_Ic[0, pack_idx].flatten()
        full_caps = system.Max_Yc[0, pack_idx].flatten()
        
        # Separate calibration and future data
        mask_calib = full_cycles <= CALIBRATION_CYCLES
        cycles_future = full_cycles[~mask_calib]
        caps_future_true = full_caps[~mask_calib]
        
        if len(cycles_future) == 0:
            print(f"No future data available for Pack {pack_idx + 1}")
            continue
        
        # Predict future capacity using fused UKF-SR approach
        initial_capacity = caps_calib[-1]  # Last calibrated capacity
        pred_curve_sr, pred_curve_fused = system.predict_with_ukf_sr_fusion(
            pack_idx, cycles_future, initial_capacity, CALIBRATION_CYCLES
        )
        
        # Calculate failure threshold
        failure_threshold = system.calculate_failure_threshold(pack_idx)
        
        # Evaluate prediction accuracy
        accuracy_metrics = system.evaluate_prediction_accuracy(
            pack_idx, cycles_future, pred_curve_fused, caps_future_true
        )
        
        # Store results
        results[pack_idx] = {
            'cycles_calib': cycles_calib,
            'caps_calib': caps_calib,
            'cycles_future': cycles_future,
            'caps_future_true': caps_future_true,
            'pred_curve_sr': pred_curve_sr,
            'pred_curve_fused': pred_curve_fused,
            'scaling_coe': scaling_coe,
            'failure_threshold': failure_threshold,
            'accuracy_metrics': accuracy_metrics
        }
        
        print(f"Accuracy Metrics for Pack {pack_idx + 1}:")
        for metric, value in accuracy_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Step 3: Generate all required visualizations
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    
    # Create visualization directory if it doesn't exist
    vis_dir = "visualizations"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Generate all required figures for each pack
    for pack_idx, result in results.items():
        pack_num = pack_idx + 1
        
        print(f"Generating visualizations for Pack {pack_num}...")
        
        # Figure 1: Overlay plot of actual vs. predicted capacity degradation trajectories
        plot_capacity_degradation_trajectory(
            result['cycles_calib'], result['caps_calib'],
            result['cycles_future'], result['caps_future_true'],
            result['pred_curve_fused'], result['failure_threshold'],
            pack_num, f'{vis_dir}/Pack_{pack_num}'
        )
        
        # Figure 2: Error analysis plot
        if len(result['pred_curve_fused']) > 0 and len(result['caps_future_true']) > 0:
            min_len = min(len(result['pred_curve_fused']), len(result['caps_future_true']))
            pred_subset = result['pred_curve_fused'][:min_len]
            true_subset = result['caps_future_true'][:min_len]
            cycles_subset = result['cycles_future'][:min_len]
            
            percentage_errors = np.abs((pred_subset - true_subset) / true_subset) * 100
            
            plot_error_analysis(cycles_subset, percentage_errors, pack_num, f'{vis_dir}/Pack_{pack_num}')
        
        # Figure 3: Scaling Coefficient visualization
        # Since we have a single scaling coefficient per pack, we'll visualize it as a horizontal line
        # But we can also show how it affects the prediction
        scaling_cycles = result['cycles_future'] if len(result['cycles_future']) > 0 else [0, 1]
        plot_scaling_coefficient(scaling_cycles, result['scaling_coe'], pack_num, f'{vis_dir}/Pack_{pack_num}')
        
        # Figure 4: Model performance metrics computed from real data
        # Calculate online accuracy and precision indices from real data
        if len(result['pred_curve_fused']) > 0 and len(result['caps_future_true']) > 0:
            min_len = min(len(result['pred_curve_fused']), len(result['caps_future_true']))
            pred_subset = result['pred_curve_fused'][:min_len]
            true_subset = result['caps_future_true'][:min_len]
            cycles_subset = result['cycles_future'][:min_len]
            
            # Calculate rolling accuracy and precision
            window_size = min(10, min_len)  # Use a small window for demonstration
            online_accuracy = []
            online_precision = []
            cycles_for_metrics = []
            
            for i in range(window_size, min_len):
                window_pred = pred_subset[i-window_size:i]
                window_true = true_subset[i-window_size:i]
                
                # Accuracy: R² score in the window
                ss_res = np.sum((window_true - window_pred) ** 2)
                ss_tot = np.sum((window_true - np.mean(window_true)) ** 2)
                acc = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Precision: 1 / (1 + RMSE) to keep it bounded
                rmse = np.sqrt(np.mean((window_pred - window_true) ** 2))
                prec = 1 / (1 + rmse)
                
                online_accuracy.append(acc)
                online_precision.append(prec)
                cycles_for_metrics.append(cycles_subset[i])
            
            if len(online_accuracy) > 0:
                plot_model_performance_metrics(
                    cycles_for_metrics, online_accuracy, online_precision, 
                    pack_num, f'{vis_dir}/Pack_{pack_num}'
                )
        
        # Figure 5: Remaining Useful Life (RUL) prediction
        # Calculate RUL based on reaching failure threshold
        if len(result['pred_curve_fused']) > 0:
            # Find the cycle when the predicted capacity drops below the failure threshold
            rul_predictions = []
            rul_ground_truth = []
            rul_cycles = []
            
            # Calculate ground truth RUL (how many cycles until failure threshold is reached)
            true_failure_cycle = None
            for i, (cycle, cap) in enumerate(zip(result['cycles_future'], result['caps_future_true'])):
                if cap <= result['failure_threshold']:
                    true_failure_cycle = cycle
                    break
            
            if true_failure_cycle is not None:
                for i, (cycle, pred_cap) in enumerate(zip(result['cycles_future'], result['pred_curve_fused'])):
                    # Estimate when the predicted capacity will reach the failure threshold
                    # This is a simplified estimation
                    if pred_cap <= result['failure_threshold']:
                        est_failure_cycle = cycle
                        current_rul = max(0, est_failure_cycle - cycle)
                    else:
                        # Linear extrapolation to estimate failure cycle
                        if i > 0:
                            degradation_rate = (result['pred_curve_fused'][i-1] - pred_cap) / (
                                result['cycles_future'][i] - result['cycles_future'][i-1]
                            ) if (result['cycles_future'][i] - result['cycles_future'][i-1]) != 0 else 0
                            
                            if degradation_rate != 0:
                                est_failure_cycle = cycle + (result['failure_threshold'] - pred_cap) / degradation_rate
                                current_rul = max(0, est_failure_cycle - cycle)
                            else:
                                current_rul = float('inf')  # No degradation detected
                        else:
                            current_rul = float('inf')
                    
                    actual_rul = max(0, true_failure_cycle - cycle) if true_failure_cycle > cycle else 0
                    
                    rul_predictions.append(current_rul)
                    rul_ground_truth.append(actual_rul)
                    rul_cycles.append(cycle)
                
                if len(rul_predictions) > 0:
                    # Add confidence bounds computed from prediction variance
                    prediction_variance = np.var(result['pred_curve_fused'])
                    rul_mean = np.mean(rul_predictions)
                    rul_upper_bound = rul_mean + np.sqrt(prediction_variance)  # Upper bound based on variance
                    rul_lower_bound = rul_mean - np.sqrt(prediction_variance)  # Lower bound based on variance
                    plot_rul_prediction(
                        rul_cycles, rul_predictions, rul_ground_truth,
                        rul_upper_bound, rul_lower_bound,
                        pack_num, f'{vis_dir}/Pack_{pack_num}'
                    )
        
        # Figure 6: Prediction uncertainty analysis
        if len(result['pred_curve_fused']) > 0 and len(result['caps_future_true']) > 0:
            min_len = min(len(result['pred_curve_fused']), len(result['caps_future_true']))
            pred_subset = result['pred_curve_fused'][:min_len]
            true_subset = result['caps_future_true'][:min_len]
            
            absolute_errors = np.abs((pred_subset - true_subset) / true_subset) * 100  # Convert to percentage
            # For prediction uncertainty, we use the variance of residuals from real data
            # Uncertainty based on actual prediction error from real data
            prediction_uncertainty = np.sqrt(np.abs((pred_subset - true_subset) / true_subset) * 100)  # Convert to percentage-based uncertainty
            
            plot_prediction_uncertainty_analysis(
                prediction_uncertainty, absolute_errors, 
                pack_num, f'{vis_dir}/Pack_{pack_num}'
            )
    
    # Print summary of all results
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for pack_idx, result in results.items():
        pack_num = pack_idx + 1
        metrics = result['accuracy_metrics']
        print(f"\nPack {pack_num}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Scaling Coefficient: {result['scaling_coe']:.4f}")
    
    print(f"\nAll visualizations saved to '{vis_dir}' directory")
    print("Comprehensive Battery Prognosis System completed successfully!")


if __name__ == "__main__":
    main()