"""
稀疏数据处理优化模块
专门处理校准数据稀疏情况下的预测优化
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
import sr as sr_module


class SparseDataAdapter:
    """
    针对稀疏数据的适配器
    当校准数据点较少时，使用特殊的处理策略
    """
    def __init__(self, universal_model, universal_scaler):
        self.model = universal_model
        self.scaler = universal_scaler
        self.best_coe = 1.0
        self.offset_param = 0.0
        self.use_sparse_strategy = False
    
    def fit_calibration_data(self, cycles_calib, capacity_calib):
        """
        针对稀疏数据的参数拟合
        """
        # 检查数据稀疏程度
        if len(cycles_calib) < 20:
            self.use_sparse_strategy = True
            print(f"  [SparseAdapter] 检测到稀疏数据 ({len(cycles_calib)} 个点)，启用稀疏策略")
            return self._fit_sparse_data(cycles_calib, capacity_calib)
        else:
            self.use_sparse_strategy = False
            # 使用常规方法
            c_proc, r_true = sr_module.preprocess_data(cycles_calib, capacity_calib)
            
            if c_proc is None or len(c_proc) < 5:
                print("  [SparseAdapter] 警告：校准数据不足，使用默认参数")
                return 1.0
            
            r_base = sr_module.predict_with_model(self.model, self.scaler, c_proc)
            
            # 简化优化目标，适用于稀疏数据
            def objective(params):
                coe, offset = params[0], params[1]
                
                if coe <= 0: return 1e10
                if abs(coe) < 1e-8: return 1e10
                
                r_pred_adjusted = (r_base / coe) + offset
                diff = r_true - r_pred_adjusted
                
                # 对于稀疏数据，使用MAE损失更稳定
                loss = np.mean(np.abs(diff))
                
                # 强正则化避免过拟合
                regularization = 0.1 * (coe - 1.0)**2 + 0.01 * offset**2
                
                return loss + regularization

            bounds = [(0.1, 10.0), (-0.5, 0.5)]  # 更严格的边界
            x0 = [1.0, 0.0]
            
            result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
            
            self.best_coe = result.x[0]
            self.offset_param = result.x[1]
            
            return self.best_coe
    
    def _fit_sparse_data(self, cycles_calib, capacity_calib):
        """
        专为稀疏数据设计的拟合方法
        """
        # 对于稀疏数据，使用更稳健的策略
        if len(capacity_calib) < 2:
            return 1.0
        
        # 计算整体退化趋势
        initial_cap = capacity_calib[0]
        final_cap = capacity_calib[-1]
        total_degradation = final_cap - initial_cap
        
        # 对于极稀疏数据，使用基于退化程度的启发式方法
        # 而不是复杂的模型匹配
        try:
            # 计算退化百分比
            degradation_percentage = abs(total_degradation) / initial_cap if initial_cap != 0 else 0
            
            # 基于退化程度设置缩放系数
            # 如果退化程度大，则目标电池退化更快，缩放系数应更大
            if degradation_percentage > 0.15:  # 退化超过15%
                coe = 2.0  # 退化非常快
            elif degradation_percentage > 0.1:  # 退化10-15%
                coe = 1.5  # 退化较快
            elif degradation_percentage > 0.05:  # 退化5-10%
                coe = 1.2  # 退化稍快
            elif degradation_percentage > 0.02:  # 退化2-5%
                coe = 1.0  # 正常退化
            else:  # 退化小于2%
                coe = 0.8  # 退化较慢
            
            # 考虑退化方向（正值表示容量增加，这在电池中不太可能）
            if total_degradation > 0:  # 如果容量增加（异常情况）
                coe = max(0.5, coe * 0.5)  # 减小系数
            
            # 限制系数范围，避免极端值
            coe = np.clip(coe, 0.1, 5.0)
            
            # 对于稀疏数据，偏移量设为0以避免过拟合
            offset = 0.0
            
            self.best_coe = coe
            self.offset_param = offset
            
            return coe
        except Exception as e:
            print(f"  [SparseAdapter] 稀疏数据拟合失败: {e}，使用默认参数")
            return 1.0
    
    def predict_trajectory(self, cycles_future, current_capacity, dt_mode='diff'):
        """
        基于稀疏数据优化的预测轨迹
        """
        # 获取基准退化率
        r_base_future = sr_module.predict_with_model(self.model, self.scaler, cycles_future)
        
        # 应用增强的尺度迁移
        r_pred_final = (r_base_future / self.best_coe) + self.offset_param
        
        # 积分还原容量
        pred_curve = [current_capacity]
        curr_cap = current_capacity
        
        for i in range(len(cycles_future)):
            # 计算 dt (时间步长)
            if i == 0:
                dt = 1.0 if len(cycles_future) == 0 else (cycles_future[1] - cycles_future[0]) if len(cycles_future) > 1 else 1.0
            else:
                dt = cycles_future[i] - cycles_future[i-1]
            
            # 物理约束：电池容量通常不会增加 (Rate <= 0)
            rate = min(r_pred_final[i], 0)
            
            dQ = rate * dt
            curr_cap += dQ
            pred_curve.append(curr_cap)
        
        # 去掉初始点，返回与 cycles_future 等长的数组
        raw_curve = np.array(pred_curve[1:])
        
        # 应用平滑处理
        if len(raw_curve) > 3:
            from scipy.interpolate import UnivariateSpline
            # 对于稀疏数据场景，使用更温和的平滑
            smooth_factor = len(cycles_future) * 0.1 if not self.use_sparse_strategy else len(cycles_future) * 0.2
            smooth_spline = UnivariateSpline(cycles_future, raw_curve, s=smooth_factor)
            smoothed_curve = smooth_spline(cycles_future)
            return smoothed_curve
        else:
            return raw_curve