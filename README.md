# DCATCN 异常检测代码仓库

本项目为论文《DCACTN：基于双向时序卷积网络与交叉注意力的无人机传感器异常检测框架》的代码实现，包含模型训练、对比实验、阈值分析及可视化脚本。

## 目录结构

```
DCATCN/                                # 根目录
├─ Flight93/                          # 基于Thor Flight93数据集的实验
│  ├─ anomalyvis/                     # 异常检测可视化模块
│  ├─ data/                           # 数据预处理与加载脚本
│  ├─ model/                          # 模型定义与加载模块
│  ├─ timecost/                       # 训练与推理时间统计脚本
│  ├─ util/                           # 工具函数（MIC筛选、SG滤波、EVT拟合等）
│  ├─ vis/                            # 绘图结果
│  ├─ AblationExperiment.py          # 消融实验：评估各模块贡献
│  ├─ ContrastExperiment.py          # 对比实验：与LSTM、CNN-BiLSTM等基线模型对比
│  ├─ GEVvis.py                       # EVT阈值拟合可视化示例
│  └─ ThresholdCompare.py             # 不同阈值策略（MSE、MAD、GEV）对比分析
```

## 环境依赖

* Python 3.10+
* TensorFlow 2.x
* tensorflow-addons
* numpy, pandas, scipy, statsmodels
* scikit-learn
* matplotlib, seaborn
* tqdm

可使用以下命令快速安装：

```bash
pip install tensorflow tensorflow-addons numpy pandas scipy statsmodels scikit-learn matplotlib seaborn tqdm
```

## 快速开始

1. **数据准备**：

   * 将 `Thor Flight93` 数据集放置于 `Flight93/data/` 目录下，确保文件命名与代码中一致。
   * 修改 `data/` 下的路径配置脚本以指向本地数据文件。

2. **预处理与特征筛选**：

   ```bash
   # 在 util/ 目录下执行预处理脚本，生成平滑后的特征和MIC筛选结果
   python util/preprocess.py
   ```

3. **模型训练与评估**：

   ```bash
   # 运行对比实验
   python Flight93/ContrastExperiment.py

   # 运行消融实验
   python Flight93/AblationExperiment.py
   ```

4. **阈值分析与可视化**：

   ```bash
   # GEV 拟合可视化
   python Flight93/GEVvis.py

   # 阈值策略对比
   python Flight93/ThresholdCompare.py
   ```

5. **结果查看**：

   * 各脚本运行后会在 `Flight93/vis/` 下生成图表和对比结果。
   * 时间开销统计文件存放于 `Flight93/timecost/`。

## 脚本说明

* `ContrastExperiment.py`：实现 CrossAttention-BiTCN 与基线模型在偏移/漂移异常场景下的检测指标对比。
* `AblationExperiment.py`：移除注意力、双向或 EVT 模块后对模型性能影响的消融分析。
* `GEVvis.py`：对训练残差极值进行 GEV 分布拟合，并绘制概率密度与阈值位置。
* `ThresholdCompare.py`：对比 MSE、MAD 和 GEV 三种阈值设定方法下的 TPR/FPR/ACC/F1-score 表现。

## 联系

如有问题或建议，请联系作者。
