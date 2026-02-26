import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# =========================================================================
# 全局可视化设置 (面向 SCI 论文的出版级质量)
# =========================================================================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = [10, 8]

# 核心颜色定义 (完全对齐 SCI 期刊标准 - 专业高级配色方案)
COLOR_HISTORY = (0.15, 0.15, 0.15)           # 历史数据 (深灰黑)
COLOR_FUTURE_TRUE = (0.0, 0.45, 0.7)         # 未来真实数据 (深蓝)
COLOR_PREDICTION = (0.835, 0.169, 0.188)     # 预测数据 (深红，类似哈佛大学红)
COLOR_FAILURE = (1.0, 0.498, 0.0)            # 失效阈值 (亮橙)
COLOR_ERROR = (0.498, 0.624, 0.8)            # 误差数据 (浅蓝)
COLOR_SCALING = (0.651, 0.337, 0.812)        # 缩放系数 (紫罗兰)
COLOR_RUL = (0.0, 0.627, 0.451)              # RUL数据 (绿松石)
COLOR_UNCERTAINTY = (0.941, 0.502, 0.502)    # 不确定性 (珊瑚红)
# =========================================================================

def plot_capacity_degradation_trajectory(cycles_calib, caps_calib, cycles_future, caps_future_true, pred_curve, failure_threshold, pack_idx, save_path=None):
    """
    Figure 1: Overlay plot of actual vs. predicted capacity degradation trajectories
    - Prediction starts from cycle 200, with model utilizing only first 200 cycles of data and pre-trained knowledge from 8 cells
    - Y-axis: Capacity (Ah), X-axis: Cycles
    - Include plot lines for: History accessible data, Future unknown data, Prediction life, Failure threshold
    """
    fig, ax = plt.subplots(figsize=(14, 6))  # Wider and flatter aspect ratio
    
    # Plot history accessible data (calibration)
    ax.plot(cycles_calib, caps_calib, 
            'o-', color=COLOR_HISTORY, markersize=4, linewidth=2, 
            label='History Accessible Data', alpha=0.7)
    
    # Plot future unknown data (ground truth)
    if len(caps_future_true) > 0:
        ax.plot(cycles_future, caps_future_true, 
                '-', color=COLOR_FUTURE_TRUE, linewidth=2.5, 
                label='Future Unknown Data (Ground Truth)', alpha=0.8)
    
    # Plot prediction trajectory
    if len(pred_curve) > 0:
        ax.plot(cycles_future, pred_curve, 
                '--', color=COLOR_PREDICTION, linewidth=2.5, 
                label='Prediction Trajectory', alpha=0.8)
    
    # Plot failure threshold
    ax.axhline(y=failure_threshold, color=COLOR_FAILURE, linestyle='-.', 
               linewidth=2.5, label=f'Failure Threshold (80% of initial)', alpha=0.9)
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Capacity (Ah)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    # Calculate the position based on data range to avoid overlap
    y_max = max(np.max(caps_calib) if len(caps_calib) > 0 else 0, 
                np.max(caps_future_true) if len(caps_future_true) > 0 else 0,
                np.max(pred_curve) if len(pred_curve) > 0 else 0,
                failure_threshold)
    y_min = min(np.min(caps_calib) if len(caps_calib) > 0 else float('inf'), 
                np.min(caps_future_true) if len(caps_future_true) > 0 else float('inf'),
                np.min(pred_curve) if len(pred_curve) > 0 else float('inf'),
                failure_threshold)
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}_Figure1_Capacity_Degradation.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_analysis(cycles, percentage_errors, pack_idx, save_path=None):
    """
    Figure 2: Error analysis plot
    - Y-axis: Percentage error (%), X-axis: Cycles
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(cycles, percentage_errors, 
            'o-', color=COLOR_ERROR, markersize=4, linewidth=2, 
            alpha=0.8, label='Percentage Error (%)')
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = np.max(percentage_errors) if len(percentage_errors) > 0 else 0
    y_min = np.min(percentage_errors) if len(percentage_errors) > 0 else 0
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}_Figure2_Error_Analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_scaling_coefficient(cycles, scaling_coefficients, pack_idx, save_path=None):
    """
    Figure 3: Scaling Coefficient visualization
    - Y-axis: Scaling Coefficient, X-axis: Cycles
    - The Scaling Coefficient represents the transfer learning bridge between cell and pack degradation patterns
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # For a single scaling coefficient, we plot it as a horizontal line
    # But we can also show how it affects the prediction over time
    ax.axhline(y=scaling_coefficients, color=COLOR_SCALING, linewidth=3, 
               label=f'Scaling Coefficient = {scaling_coefficients:.4f}', alpha=0.8)
    
    # Optionally, show how the coefficient varies over time if we have multiple values
    if hasattr(scaling_coefficients, '__len__') and len(scaling_coefficients) > 1:
        ax.plot(cycles, scaling_coefficients, 
                'o-', color=COLOR_SCALING, markersize=5, linewidth=2, 
                label='Scaling Coefficient Evolution', alpha=0.8)
    else:
        ax.plot([cycles[0], cycles[-1]], [scaling_coefficients, scaling_coefficients], 
                'o-', color=COLOR_SCALING, markersize=5, linewidth=3, 
                label=f'Scaling Coefficient = {scaling_coefficients:.4f}', alpha=0.8)
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Scaling Coefficient', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    if hasattr(scaling_coefficients, '__len__') and len(scaling_coefficients) > 1:
        y_max = np.max(scaling_coefficients)
        y_min = np.min(scaling_coefficients)
    else:
        y_max = scaling_coefficients
        y_min = scaling_coefficients
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_Figure3_Scaling_Coefficient.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_Figure3_Scaling_Coefficient.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_performance_metrics(cycles, accuracy_indices, precision_indices, pack_idx, save_path=None):
    """
    Figure 4: Model performance metrics
    - Y-axis: Online Accuracy Index and Online Precision Index, X-axis: Cycles
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(cycles, accuracy_indices, 
            '-', color=COLOR_HISTORY, linewidth=2.5, label='Online Accuracy Index', alpha=0.8)
    ax.plot(cycles, precision_indices, 
            '--', color=COLOR_PREDICTION, linewidth=2.5, label='Online Precision Index', alpha=0.8)
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Index Value', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = max(np.max(accuracy_indices) if len(accuracy_indices) > 0 else 0,
                np.max(precision_indices) if len(precision_indices) > 0 else 0)
    y_min = min(np.min(accuracy_indices) if len(accuracy_indices) > 0 else 0,
                np.min(precision_indices) if len(precision_indices) > 0 else 0)
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_Figure4_Performance_Metrics.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_Figure4_Performance_Metrics.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_rul_prediction(cycles, rul_predictions, rul_ground_truth, rul_upper_bound=None, rul_lower_bound=None, pack_idx=1, save_path=None):
    """
    Figure 5: Remaining Useful Life (RUL) prediction
    - Y-axis: RUL, X-axis: Cycles
    - Include plot lines for: Mean Prediction, Accuracy Limit, and Ground Truth
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot RUL predictions
    ax.plot(cycles, rul_predictions, 
            'o-', color=COLOR_RUL, markersize=4, linewidth=2.5, 
            label='Mean Prediction', alpha=0.8)
    
    # Plot ground truth RUL
    ax.plot(cycles, rul_ground_truth, 
            's-', color=COLOR_FUTURE_TRUE, markersize=4, linewidth=2.5, 
            label='Ground Truth', alpha=0.8)
    
    # Plot accuracy limits if provided
    if rul_upper_bound is not None and rul_lower_bound is not None:
        ax.fill_between(cycles, rul_lower_bound, rul_upper_bound, 
                       alpha=0.2, color=COLOR_RUL, label='Accuracy Limit')
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('RUL', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = max(np.max(rul_predictions) if len(rul_predictions) > 0 else 0,
                np.max(rul_ground_truth) if len(rul_ground_truth) > 0 else 0)
    y_min = min(np.min(rul_predictions) if len(rul_predictions) > 0 else 0,
                np.min(rul_ground_truth) if len(rul_ground_truth) > 0 else 0)
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_Figure5_RUL_Prediction.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_Figure5_RUL_Prediction.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_uncertainty_analysis(uncertainty_values, error_values, pack_idx, save_path=None):
    """
    Figure 6: Prediction uncertainty analysis
    - Y-axis: Absolute error (%), X-axis: Prediction Uncertainty (%)
    - Format as a scatter plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(uncertainty_values, error_values, 
                         c=error_values, cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5, label='Uncertainty vs Error')
    
    ax.set_xlabel('Prediction Uncertainty (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Absolute Error (%)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error Intensity', fontsize=14)
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = np.max(error_values) if len(error_values) > 0 else 0
    y_min = np.min(error_values) if len(error_values) > 0 else 0
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_Figure6_Uncertainty_Analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}_Figure6_Uncertainty_Analysis.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot_accuracy_change_curve(cycles, percentage_errors, pack_idx, save_path=None):
    """
    New Figure: Accuracy change curve showing model accuracy over time
    - X-axis: Cycles
    - Y-axis: Percentage Error (%)
    - Shows how prediction accuracy changes over the cycles
    """
    fig, ax = plt.subplots(figsize=(14, 6))  # Wide aspect ratio
    
    ax.plot(cycles, percentage_errors, 
            'o-', color=COLOR_ERROR, markersize=4, linewidth=2, 
            label='Percentage Error (%)', alpha=0.8)
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = np.max(percentage_errors) if len(percentage_errors) > 0 else 0
    y_min = np.min(percentage_errors) if len(percentage_errors) > 0 else 0
    
    # Add pack label at specified position
    ax.text(0.02, 0.95, f'Pack-{pack_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}_Accuracy_Change_Curve.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_results(c_calib, cap_calib, c_future, cap_true, cap_pred, coe, target_idx, save_path=None):
    """
    Plotting function from main.py - plots capacity degradation results
    - Historical data, future ground truth, and predicted trajectory
    - With failure threshold and Pack identifier
    """
    from sklearn.metrics import mean_squared_error  # Import here to avoid circular dependency
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 8)) # 配置输出的绘图窗口的尺寸
    
    # 绘制历史数据
    plt.plot(c_calib, cap_calib, 'k.', markersize=8, label='Accessible Data', alpha=0.6)
    
    # 绘制真实未来
    plt.plot(c_future, cap_true, 'g-', linewidth=2, label='Ground Truth')
    
    # 绘制预测未来
    plt.plot(c_future, cap_pred, 'r--', linewidth=3, label=f'Predicted Trajectory')
    
    # 待预测组初始容量 = 校准阶段最后一个容量值（预测起始点）
    pred_initial_cap = cap_calib[-1]
    half_pred_initial_cap = pred_initial_cap * 0.5  # 50%阈值
    # 绘制50%阈值线
    plt.axhline(
        y=half_pred_initial_cap, 
        color='orange', 
        linestyle=':', 
        linewidth=2,
        label=f'Failure Threshold'
    )

    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = max(np.max(cap_calib) if len(cap_calib) > 0 else 0,
                np.max(cap_true) if len(cap_true) > 0 else 0,
                np.max(cap_pred) if len(cap_pred) > 0 else 0)
    y_min = min(np.min(cap_calib) if len(cap_calib) > 0 else float('inf'),
                np.min(cap_true) if len(cap_true) > 0 else float('inf'),
                np.min(cap_pred) if len(cap_pred) > 0 else float('inf'))

    # Calculate a position that avoids overlapping with data curves
    # Place the text in the bottom-left corner, adjusted to avoid overlap
    plt.text(0.02, 0.05, f"Pack {target_idx + 1}", 
             transform=plt.gca().transAxes, 
             fontsize=20,           # 字号大小，可根据需要调整
             # fontweight='bold',     # 加粗
             color='black',         # 字体颜色
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none') # 可选：加个半透明白底防止遮挡线条
    )

    # 计算误差
    rmse = np.sqrt(mean_squared_error(cap_true, cap_pred))
    
    # 设置绘图字体字号
    plt.xlabel("Cycles", fontsize=24)
    plt.ylabel("Capacity (mAh)", fontsize=24)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, None)
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_capacity_curves(Max_Ic, Max_Yc, save_path=None):
    """
    New Figure: 2D overlay curve chart showing real capacity degradation trends for all cells and packs
    - X-axis: Cycles
    - Y-axis: Capacity (Ah)
    - Shows capacity degradation curves for different cells and packs
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors for different groups
    num_cells = 8  # Cells 1-8
    num_packs = 6  # Packs 9-14
    
    # Create color maps for cells and packs
    cell_colors = plt.cm.Blues(np.linspace(0.3, 0.9, num_cells))
    pack_colors = plt.cm.Reds(np.linspace(0.3, 0.9, num_packs))
    
    # Track min/max values for dynamic label placement
    all_capacities = []
    
    # Plot cell curves (1-8)
    for i in range(num_cells):
        cycles = Max_Ic[0, i].flatten()
        capacities = Max_Yc[0, i].flatten()
        ax.plot(cycles, capacities, 
                color=cell_colors[i], 
                linewidth=2, 
                label=f'Cell {i+1}',
                alpha=0.8)
        all_capacities.extend(capacities)
    
    # Plot pack curves (9-14)
    for i in range(num_cells, num_cells + num_packs):
        cycles = Max_Ic[0, i].flatten()
        capacities = Max_Yc[0, i].flatten()
        ax.plot(cycles, capacities, 
                color=pack_colors[i-num_cells], 
                linewidth=2, 
                label=f'Pack {i+1}',
                alpha=0.8)
        all_capacities.extend(capacities)
    
    ax.set_xlabel('Cycles', fontsize=16, fontweight='bold')
    ax.set_ylabel('Capacity (Ah)', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    if all_capacities:
        y_max = max(all_capacities)
        y_min = min(all_capacities)
    else:
        y_max, y_min = 1, 0
    
    # Place the text in the top-left corner, adjusted to avoid overlap
    ax.text(0.02, 0.95, 'All Packs & Cells', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}_All_Capacity_Curves.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_rul_trajectory(t_I, RUL_mid, RUL_low, RUL_high, RUL_true, target_idx, save_path=None):
    """
    1. 绘制带有 95% 置信区间的 RUL 预测轨迹图 (对应 MATLAB Figure 11111)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制 95% 置信区间
    ax.fill_between(t_I, RUL_low, RUL_high, color=COLOR_RUL, alpha=0.2, edgecolor='none', label='95% Confidence Interval')
    
    # 绘制预测均值
    ax.plot(t_I, RUL_mid, color=COLOR_RUL, linewidth=2.5, label='Predicted result')
    
    # 绘制真实失效水平线
    ax.axhline(y=RUL_true, color=COLOR_FUTURE_TRUE, linestyle='--', linewidth=2.5, label='Real failure cycle')
    
    ax.set_xlabel('Prediction start time (Cycles)')
    ax.set_ylabel('Predicted life (Cycles)')
    ax.grid(True, alpha=0.15)
    ax.legend(loc='lower right', frameon=False)
    
    # Dynamic position calculation for Pack label to avoid overlapping with data or legend
    y_max = max(np.max(RUL_mid) if len(RUL_mid) > 0 else 0,
                np.max(RUL_low) if len(RUL_low) > 0 else 0,
                np.max(RUL_high) if len(RUL_high) > 0 else 0,
                RUL_true)
    y_min = min(np.min(RUL_mid) if len(RUL_mid) > 0 else float('inf'),
                np.min(RUL_low) if len(RUL_low) > 0 else float('inf'),
                np.min(RUL_high) if len(RUL_high) > 0 else float('inf'),
                RUL_true)
    
    # 添加标签标记 at specified position
    ax.text(0.02, 0.95, f'Pack-{target_idx}', transform=ax.transAxes, fontsize=16, 
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if save_path:
        # Ensure directory exists
        import os
        directory = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f"{save_path}_RUL_Trajectory.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_pdf_evolution(t_I, loc_list, Pr_list, RUL_true, target_idx, step=5, save_path=None):
    """
    2. RUL 概率密度 3D 演化图 (对应 MATLAB 高级可视化 1)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    verts = []
    facecolors = []
    
    max_prob = 0
    # 降采样以避免 3D 图过于密集
    for i in range(0, len(t_I), step):
        if i >= len(Pr_list) or len(Pr_list[i]) == 0:
            continue
            
        locs = np.array(loc_list[i])
        pdfs = np.array(Pr_list[i])
        
        if np.max(pdfs) > 0:
            pdfs = pdfs / np.max(pdfs) * 100  # 归一化
            
        max_prob = max(max_prob, np.max(pdfs))
        
        # 为了使用 PolyCollection 填充颜色，需要闭合多边形 (将首尾概率置为0)
        locs_poly = np.concatenate(([locs[0]], locs, [locs[-1]]))
        pdfs_poly = np.concatenate(([0], pdfs, [0]))
        verts.append(list(zip(locs_poly, pdfs_poly)))
        
        # 绘制黑色的轮廓线
        ax.plot(locs, np.full_like(locs, t_I[i]), pdfs, color=(0.1, 0.1, 0.1, 0.5), linewidth=1, zdir='y')
    
    # 使用 PolyCollection 绘制带透明度的填充面
    poly = PolyCollection(verts, facecolors=COLOR_PREDICTION, alpha=0.3, edgecolors='none')
    ax.add_collection3d(poly, zs=t_I[::step][:len(verts)], zdir='y')
    
    # 绘制真实失效的参考平面线
    ax.plot([RUL_true, RUL_true], [t_I[0], t_I[-1]], [0, max_prob], color='red', linestyle='--', linewidth=2, label='True RUL')
    
    ax.set_xlabel('Predicted RUL (Cycle)', labelpad=15)
    ax.set_ylabel('Current Time (Cycle)', labelpad=15)
    ax.set_zlabel('Probability Density (Scaled)', labelpad=15)
    ax.set_title('Evolution of RUL Probability Density Function', pad=20)
    
    ax.view_init(elev=30, azim=-45)
    ax.grid(True, alpha=0.3)
    
    # 动态调整轴范围
    ax.set_xlim(np.min(loc_list), np.max(loc_list))
    ax.set_ylim(t_I[0], t_I[-1])
    ax.set_zlim(0, max_prob * 1.05)
    
    if save_path:
        plt.savefig(f"{save_path}_3D_PDF.pdf")
    plt.show()

def plot_joint_pdf(t_I, failure_grid, joint_pdf_matrix, RUL_true, target_idx, save_path=None):
    """
    3. 联合概率密度热力图 (对应 MATLAB 核心修复的 TTF Joint PDF)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    T, F = np.meshgrid(t_I, failure_grid)
    
    # 使用 contourf 绘制平滑的概率分布等高线填充
    cf = ax.contourf(T, F, joint_pdf_matrix, levels=50, cmap='jet', alpha=0.9, linestyles='none')
    
    # 绘制真实失效寿命线
    ax.axhline(y=RUL_true, color='white', linestyle='--', linewidth=2.5, label='True Failure')
    
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('Probability Density (Normalized)')
    
    ax.set_xlabel('Prediction Time (Cycle)')
    ax.set_ylabel('Predicted Failure Time (Cycle)')
    ax.set_title('Joint Probability Density of Time-to-Failure')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2, color='white')
    
    if save_path:
        plt.savefig(f"{save_path}_Joint_PDF.pdf")
    plt.show()

def plot_error_confidence_pareto(t_I, RUL_mid, RUL_low, RUL_high, RUL_true, save_path=None):
    """
    4. 误差-置信度 Pareto 散点图
    """
    # 过滤无效数据
    valid = (RUL_mid > 0) & (RUL_high > RUL_low)
    RUL_mid = RUL_mid[valid]
    RUL_low = RUL_low[valid]
    RUL_high = RUL_high[valid]
    cycles = t_I[valid]
    
    # 计算绝对误差百分比和置信区间宽度百分比
    rel_err = np.abs(RUL_mid - RUL_true) / RUL_true * 100
    conf_width = (RUL_high - RUL_low) / (RUL_mid + 1e-9) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(conf_width, rel_err, c=cycles, cmap='Blues_r', 
                         s=80, alpha=0.8, edgecolors='gray', linewidth=0.5)
    
    ax.set_xlabel('Prediction Uncertainty (%)')
    ax.set_ylabel('Absolute Error (%)')
    ax.set_title('Error vs. Uncertainty Pareto Front')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Prediction Start Cycle')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(f"{save_path}_Pareto.pdf")
    plt.show()


