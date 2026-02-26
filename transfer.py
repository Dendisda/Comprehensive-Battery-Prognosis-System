import numpy as np
from scipy.optimize import minimize
import sr as sr_module # 仅用于调用预处理，保证数据处理一致性

class ScaleAdapter:
    def __init__(self, universal_model, universal_scaler):
        self.model = universal_model
        self.scaler = universal_scaler
        self.best_coe = 1.0
        
    def fit_calibration_data(self, cycles_calib, capacity_calib):
        """
        计算最佳迁移系数 Coe (对应 MATLAB 中的 Transfer.m 逻辑)
        """
        # 1. 获取目标电池真实的退化率
        c_proc, r_true = sr_module.preprocess_data(cycles_calib, capacity_calib)
        
        if c_proc is None or len(c_proc) < 5:
            print("  [Transfer] 警告：校准数据不足，无法进行迁移优化")
            return 1.0
        
        # 2. 获取通用模型在这些时刻的"基准预测"
        r_base = sr_module.predict_with_model(self.model, self.scaler, c_proc)
        
        # 3. 定义目标函数 - 使用更稳定的优化目标
        def objective(coe):
            # 物理约束：Coe 必须是正数。
            if coe[0] <= 0: return 1e10
            
            # 避免除零错误
            if abs(coe[0]) < 1e-8: return 1e10
            
            r_pred_scaled = r_base / coe[0]
            
            # 使用改进的损失函数，结合多种损失
            diff = r_true - r_pred_scaled
            
            # MAE (L1) 损失 - 对异常值更鲁棒
            mae_loss = np.mean(np.abs(diff))
            
            # MSE (L2) 损失 - 对小误差敏感
            mse_loss = np.mean(diff ** 2)
            
            # 组合损失函数，平衡鲁棒性和精确性
            combined_loss = 0.6 * mae_loss + 0.4 * mse_loss
            
            # 添加正则化项以避免过拟合
            regularization = 0.005 * (coe[0] - 1.0)**2
            
            return combined_loss + regularization

        # 4. 执行优化 (L-BFGS-B 支持边界约束)
        # 边界设为 [0.01, 50]，允许更大的个体差异
        result = minimize(objective, x0=[1.0], bounds=[(0.01, 50.0)], method='L-BFGS-B')
        
        self.best_coe = result.x[0]
        return self.best_coe

    def predict_trajectory(self, cycles_future, current_capacity, dt_mode='diff'):
        """
        基于迁移系数预测未来轨迹 (对应 MATLAB 中的 Prognosis.m 逻辑)
        """
        # 1. 获取基准退化率
        r_base_future = sr_module.predict_with_model(self.model, self.scaler, cycles_future)
        
        # 2. 应用尺度迁移：Rate_target = Rate_universal / Coe
        r_pred_final = r_base_future / self.best_coe
        
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
            # 如果 SR 预测出了正数（充电），强制置为 0 或保留微小噪音
            rate = min(r_pred_final[i], 0)
            
            dQ = rate * dt
            curr_cap += dQ
            pred_curve.append(curr_cap)
            
        # 去掉初始点，返回与 cycles_future 等长的数组
        return np.array(pred_curve[1:])