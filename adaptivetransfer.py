import numpy as np
from scipy.optimize import minimize
import sr as sr_module


class AdaptiveScaleAdapter:
    """
    自适应缩放适配器，根据数据特征动态调整参数
    增强版迁移学习模块，不仅使用缩放系数，还考虑偏移和其他适应性参数
    """
    def __init__(self, universal_model, universal_scaler):
        self.model = universal_model
        self.scaler = universal_scaler
        self.best_coe = 1.0  # 缩放系数
        self.offset_param = 0.0  # 偌移参数
        self.adaptation_params = {}  # 存储其他适应性参数
        
    def fit_calibration_data(self, cycles_calib, capacity_calib):
        """
        根据数据特征自适应计算最佳迁移参数，包括缩放系数和偏移参数
        实现更丰富的知识迁移策略
        """
        # 1. 获取目标电池真实的退化率
        c_proc, r_true = sr_module.preprocess_data(cycles_calib, capacity_calib)
        
        if c_proc is None or len(c_proc) < 5:
            print("  [AdaptiveTransfer] 警告：校准数据不足，无法进行迁移优化")
            return 1.0
        
        # 2. 分析数据特征以确定最优参数
        data_characteristics = self.analyze_data_characteristics(cycles_calib, capacity_calib)
        
        # 3. 获取通用模型在这些时刻的"基准预测"
        r_base = sr_module.predict_with_model(self.model, self.scaler, c_proc)
        
        # 4. 定义复合目标函数，同时优化缩放系数和偏移参数
        def objective(params):
            coe, offset = params[0], params[1]
            
            if coe <= 0: return 1e10
            if abs(coe) < 1e-8: return 1e10
            
            # 应用缩放和偏移
            r_pred_adjusted = (r_base / coe) + offset
            diff = r_true - r_pred_adjusted
            
            # 根据数据特征选择不同的损失函数
            if data_characteristics['noise_level'] > 0.5:  # 高噪声
                # 使用更鲁棒的MAE损失
                loss = np.mean(np.abs(diff))
            else:  # 低噪声
                # 使用MSE损失以获得更高精度
                loss = np.mean(diff ** 2)
            
            # 添加正则化项，避免参数过大
            regularization = 0.01 * (coe - 1.0)**2 + 0.001 * offset**2
            
            return loss + regularization

        # 5. 设置优化边界
        coe_bounds = self.get_optimization_bounds(data_characteristics)[0]
        offset_bounds = (-1.0, 1.0)  # 偌移参数的合理范围
        
        bounds = [coe_bounds, offset_bounds]
        
        # 初始参数猜测
        x0 = [1.0, 0.0]
        
        # 6. 执行优化
        result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        
        self.best_coe = result.x[0]
        self.offset_param = result.x[1]
        
        # 存储额外的适应性参数
        self.adaptation_params = {
            'data_noise_level': data_characteristics['noise_level'],
            'avg_change_rate': data_characteristics['avg_change_rate'],
            'optimization_success': result.success,
            'final_cost': result.fun
        }
        
        return self.best_coe
    
    def analyze_data_characteristics(self, cycles, capacities):
        """
        分析数据特征以指导参数选择
        """
        # 计算容量变化率
        if len(capacities) > 1:
            capacity_changes = np.diff(capacities)
            avg_change_rate = np.mean(np.abs(capacity_changes))
            
            # 计算噪声水平（通过局部变化的稳定性）
            if len(capacity_changes) > 2:
                local_changes_std = np.std(capacity_changes)
                noise_level = local_changes_std / (abs(avg_change_rate) + 1e-8)
            else:
                noise_level = 0.1
        else:
            avg_change_rate = 0
            noise_level = 0.1
        
        # 根据噪声水平确定正则化强度
        if noise_level > 0.5:
            regularization_factor = 0.05  # 高噪声需要更强正则化
        else:
            regularization_factor = 0.01  # 低噪声可以使用较弱正则化
        
        return {
            'avg_change_rate': avg_change_rate,
            'noise_level': min(noise_level, 2.0),  # 限制最大噪声水平
            'regularization_factor': regularization_factor
        }
    
    def get_optimization_bounds(self, data_characteristics):
        """
        根据数据特征设置优化边界
        """
        noise_level = data_characteristics['noise_level']
        
        if noise_level > 0.8:
            # 高噪声数据使用更宽的边界
            return [(0.05, 30.0)]
        elif noise_level > 0.5:
            # 中等噪声数据使用中等边界
            return [(0.1, 20.0)]
        else:
            # 低噪声数据使用较窄的边界
            return [(0.2, 10.0)]

    def predict_trajectory(self, cycles_future, current_capacity, dt_mode='diff'):
        """
        基于增强的迁移学习策略预测未来轨迹，使用平滑插值方法
        """
        # 1. 获取基准退化率
        r_base_future = sr_module.predict_with_model(self.model, self.scaler, cycles_future)
        
        # 2. 应用增强的尺度迁移：Rate_target = (Rate_universal / Coe) + Offset
        r_pred_final = (r_base_future / self.best_coe) + self.offset_param
        
        # 3. 积分还原容量
        pred_curve = [current_capacity]
        curr_cap = current_capacity
        
        for i in range(len(cycles_future)):
            # 计算 dt (时间步长)
            if i == 0:
                dt = 1.0 # 默认第一步
            else:
                dt = cycles_future[i] - cycles_future[i-1]
            
            # 物理约束：电池容量通常不会增加 (Rate <= 0)
            # 但要考虑offset可能使rate变为正数的情况
            rate = min(r_pred_final[i], 0)
            
            dQ = rate * dt
            curr_cap += dQ
            pred_curve.append(curr_cap)
        
        # 去掉初始点，返回与 cycles_future 等长的数组
        raw_curve = np.array(pred_curve[1:])
        
        # 4. 应用平滑处理以获得平滑曲线
        if len(raw_curve) > 3:  # 确保有足够的点进行平滑
            from scipy.interpolate import UnivariateSpline
            # 使用样条插值进行平滑，根据数据特征调整平滑度
            smooth_factor = len(cycles_future) * 0.05  # 根据数据点数量调整平滑度
            smooth_spline = UnivariateSpline(cycles_future, raw_curve, s=smooth_factor)
            smoothed_curve = smooth_spline(cycles_future)
            return smoothed_curve
        else:
            return raw_curve