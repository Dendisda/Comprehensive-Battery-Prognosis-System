import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# === 导入自定义模块 ===
import sr as sr_module
import transfer as tl_module

# ================= 配置区域 =================
MAT_FILE_PATH = 'best_estimation.mat'
TRAIN_CELL_INDICES = list(range(8))    # 源域 (Cell 1-8)
TARGET_CELL_INDEX = 8                  # 目标域 (Python索引 9 对应目标 pack10 )
CALIBRATION_CYCLES = 200               # 前 N 圈用于计算 Coe
# ===========================================

class BatteryPrognosisSystem:
    def __init__(self, mat_file):
        self.mat_file = mat_file
        self.universal_model = None
        self.universal_scaler = None
        self.load_data()

    def load_data(self):
        print(f"--- 正在加载数据: {self.mat_file} ---")
        try:
            data = sio.loadmat(self.mat_file)
            self.Max_Ic = data['Max_Ic']
            self.Max_Yc = data['Max_Yc']
            print("数据加载成功。")
        except FileNotFoundError:
            print("错误：找不到 .mat 文件")
            exit()

    def run_training_phase(self, train_indices):
        """
        阶段 1: 形状提取 (Shape Extraction)
        """
        print("\n=== 阶段 1: 提取通用退化模式 (Training) ===")
        
        # 准备数据列表
        all_cycles = [self.Max_Ic[0, i].flatten() for i in train_indices]
        all_caps = [self.Max_Yc[0, i].flatten() for i in train_indices]
        
        # 调用 control_sr 进行训练
        gp, scaler, score, formula = sr_module.train_universal_sr_model(all_cycles, all_caps)
        
        if gp:
            self.universal_model = gp
            self.universal_scaler = scaler
            print(f"  -> 训练完成 | 拟合度 R2: {score:.4f}")
            print(f"  -> 发现的物理公式: Rate = {formula}")
        else:
            print("  -> 训练失败")

    def run_prediction_phase(self, target_idx, calib_limit):
        """
        阶段 2 & 3: 尺度自适应与预测
        """
        print(f"\n=== 阶段 2: 目标域迁移预测 (Cell {target_idx+1}) ===")
        
        if self.universal_model is None:
            print("错误：请先运行训练阶段。")
            return

        # 1. 准备目标数据
        full_cycles = self.Max_Ic[0, target_idx].flatten()
        full_caps = self.Max_Yc[0, target_idx].flatten()
        
        # 切分已知(Calibration)与未知(Future)
        mask_calib = full_cycles <= calib_limit
        cycles_calib = full_cycles[mask_calib]
        caps_calib = full_caps[mask_calib]
        
        cycles_future = full_cycles[~mask_calib]
        caps_future_true = full_caps[~mask_calib]
        
        # 2. 初始化迁移适配器 (Transfer Adapter)
        adapter = tl_module.ScaleAdapter(self.universal_model, self.universal_scaler)
        
        # 3. 计算迁移系数 (Coe)
        best_coe = adapter.fit_calibration_data(cycles_calib, caps_calib)
        print(f"  -> 优化得到的缩放系数 (Coe): {best_coe:.4f}")
        if best_coe > 1:
            print("     (解释: 目标电池比通用模型衰减更慢/更耐用)")
        else:
            print("     (解释: 目标电池比通用模型衰减更快)")

        # 4. 预测未来
        print("\n=== 阶段 3: 生成预测轨迹 ===")
        current_cap = caps_calib[-1]
        pred_curve = adapter.predict_trajectory(cycles_future, current_cap)
        
        # 5. 可视化与评估
        # 导入绘图模块
        import draw
        # 使用统一的绘图接口
        draw.plot_results(cycles_calib, caps_calib, cycles_future, caps_future_true, pred_curve, best_coe, target_idx)

if __name__ == "__main__":
    system = BatteryPrognosisSystem(MAT_FILE_PATH)
    
    # 1. 训练通用模型
    system.run_training_phase(TRAIN_CELL_INDICES)
    
    # 2. 对目标电池进行迁移学习和预测
    system.run_prediction_phase(TARGET_CELL_INDEX, CALIBRATION_CYCLES)