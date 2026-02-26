import numpy as np
from gplearn.genetic import SymbolicRegressor
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# === SR 参数配置  ===
SR_PARAMS = {
    'population_size': 800,           # 进一步降低种群规模，减少过拟合风险
    'generations': 300,               # 增加进化代数，提高优化效果
    'parsimony_coefficient': 0.2,     # 增加复杂度惩罚，获得更简洁的表达式
    'function_set': ('add', 'sub', 'mul', 'sqrt'), # 进一步简化函数集，避免过度复杂
    'verbose': 1,                     # 启用详细输出，便于调试
    'max_samples': 0.85,
    'random_state': 42,
    'n_jobs': 1
}
# 重要参数：放大因子
# 将微小的退化率 (e.g. -0.0001) 放大到 SR 敏感的范围 (e.g. -1.0)
RATE_SCALING_FACTOR = 800.0         # 进一步调整放大因子，平衡敏感性和稳定性
# 该参量十分敏感



def preprocess_data(cycles, capacity, smooth_factor=0.15):  # 降低平滑因子以减少过度平滑
    try:
        if len(cycles) < 8:  # 降低最小数据要求
            return None, None
            
        # 异常值检测和处理
        capacity = remove_outliers(capacity)
        
        # 使用更稳健的平滑方法
        spline = UnivariateSpline(cycles, capacity, s=smooth_factor * len(cycles), k=min(3, len(cycles)-1))
        smoothed_cap = spline(cycles)
        
        # 自适应步长选择：基于数据点数量动态调整
        if len(cycles) < 30:
            step = max(2, len(cycles) // 20)  # 对于少数据点使用较小步长
        elif len(cycles) < 100:
            step = max(3, len(cycles) // 15)  # 对于中等数据点使用适中步长
        else:
            step = max(4, len(cycles) // 25)  # 对于多数据点使用适中步长
        
        # 防止越界
        if len(cycles) <= step + 1:
            return None, None
            
        # 计算退化率，使用中心差分以提高精度
        rate = np.zeros(len(cycles) - step)
        cycle_points = np.zeros(len(cycles) - step)
        
        for i in range(len(rate)):
            idx_now = i
            idx_future = i + step
            delta_cap = smoothed_cap[idx_future] - smoothed_cap[idx_now]
            delta_cycle = cycles[idx_future] - cycles[idx_now]
            if delta_cycle != 0:
                rate[i] = delta_cap / delta_cycle
            else:
                rate[i] = 0
            cycle_points[i] = cycles[idx_now]
        
        # 使用更严格的过滤，但保留更多有效数据
        Q1 = np.percentile(rate, 25)
        Q3 = np.percentile(rate, 75)
        IQR = Q3 - Q1
        
        # 物理约束：电池容量通常只减少，且退化率不应过大
        mask_physical = (rate <= 0) & (rate > -5.0)  # 适当放宽退化率限制
        # 统计过滤：使用四分位数范围，但更加严格
        mask_stat = (rate > Q1 - 1.5 * IQR) & (rate < Q3 + 1.5 * IQR)  # 使用标准IQR倍数
        mask = mask_physical & mask_stat
        
        if np.sum(mask) < 3:  # 确保有足够的有效数据点
            print(f"  [Warning] Insufficient valid data after filtering! Valid points: {np.sum(mask)}, Raw rate mean: {np.mean(rate):.6f}")
            # 如果数据点太少，尝试放宽过滤条件
            mask_loose = mask_physical  # 只使用物理约束
            if np.sum(mask_loose) >= 3:
                print("  [Info] Using relaxed filtering criteria")
                return cycle_points[mask_loose], rate[mask_loose]
            else:
                return None, None
            
        return cycle_points[mask], rate[mask]
        
    except Exception as e:
        print(f"Preprocess warning: {e}")
        return None, None


def remove_outliers(data, threshold=2.0):
    """
    使用Z-score方法移除异常值
    """
    data = np.array(data)
    z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))  # 添加小值避免除零
    filtered_data = np.where(z_scores <= threshold, data, np.median(data))
    return filtered_data

def get_features(cycles):
    """
    [保持原样] 特征工程
    """
    c_safe = np.maximum(cycles, 1e-6)
    X_raw = cycles.reshape(-1, 1)
    
    X_poly = np.column_stack((
        X_raw, 
        X_raw**2, 
        np.log(c_safe),
        np.sqrt(c_safe)
    ))
    return X_poly

def train_universal_sr_model(all_cycles_list, all_capacity_list):
    """
    [修改] 训练通用模型，但在 fit 前应用放大因子，并加入交叉验证评估
    """
    X_all = []
    y_all = []
    
    print("  -> [SR] 正在预处理并合并源域数据 (Spline)...")
    for cycles, cap in zip(all_cycles_list, all_capacity_list):
        c_clean, r_clean = preprocess_data(cycles, cap)
        if c_clean is not None and len(c_clean) > 5:  # 进一步降低数据点要求
            feat = get_features(c_clean)
            X_all.append(feat)
            y_all.append(r_clean)
            
    if not X_all:
        print("错误：没有足够的有效训练数据")
        return None, None, 0, None

    # 垂直拼接所有数据
    X_stacked = np.vstack(X_all)
    y_stacked = np.concatenate(y_all)
    
    # 检查数据有效性
    if len(y_stacked) == 0 or np.all(np.isnan(y_stacked)) or np.all(np.isinf(y_stacked)):
        print("错误：没有有效的训练数据或数据包含NaN/Inf值")
        return None, None, 0, None
    
    # 归一化输入
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stacked)
    
    # 【核心修改步骤】
    # 放大目标值 (Target Scaling)
    y_scaled = y_stacked * RATE_SCALING_FACTOR
    
    # 检查放大后的数据有效性
    if np.any(np.isnan(y_scaled)) or np.any(np.isinf(y_scaled)):
        print("警告：放大后的数据包含NaN或Inf值，正在进行清理")
        valid_mask = np.isfinite(y_scaled)
        X_scaled = X_scaled[valid_mask]
        y_scaled = y_scaled[valid_mask]
    
    print(f"  -> [SR] 开始训练 (样本数: {len(y_scaled)})...")
    print(f"  -> [SR] 目标值范围 (放大后): [{np.min(y_scaled):.4f}, {np.max(y_scaled):.4f}]")
    
    # 创建模型实例
    gp = SymbolicRegressor(**SR_PARAMS)
    
    # 在训练前进行简单的交叉验证评估
    from sklearn.model_selection import cross_val_score
    try:
        cv_scores = cross_val_score(gp, X_scaled, y_scaled, cv=5, scoring='r2')
        print(f"  -> [SR] 5折交叉验证平均R²得分: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    except Exception as e:
        print(f"  -> [SR] 交叉验证出现异常: {str(e)}")
    
    # 训练模型
    gp.fit(X_scaled, y_scaled)
    
    # 计算训练集上的得分
    score = gp.score(X_scaled, y_scaled)
    
    # 输出发现的公式
    formula = str(gp._program)
    print(f"  -> [SR] 发现的符号回归公式: {formula}")
    
    return gp, scaler, score, formula

def predict_with_model(model, scaler, cycles_input):
    """
    [修改] 预测接口，预测后除以放大因子
    """
    X_poly = get_features(cycles_input)
    X_scaled = scaler.transform(X_poly)
    
    # 模型输出的是放大的值
    pred_scaled = model.predict(X_scaled)
    
    # 【核心修改步骤】
    # 还原回真实的物理量级
    return pred_scaled / RATE_SCALING_FACTOR