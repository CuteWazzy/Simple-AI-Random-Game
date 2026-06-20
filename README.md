# AI 数字对战游戏 - 训练系统

## 项目简介

基于 C++ 数字对战游戏代码，用 Python + PyTorch 实现的 AI 训练系统。
包含游戏引擎、神经网络模型、REINFORCE 训练、遗传算法多种群并行训练、
可视化训练、人机对弈等完整功能。

## 游戏规则

- 4 位玩家，每人初始 HP=10
- 每回合当前玩家选择目标，抽 0~999999 的随机数
- 命中数 = 数字中含 (6,9)/(7,8)/(9,1) 两位子序列的个数
- 5 位数命中翻倍；4 位选道具；3 位回 2 血；2 位回 4 血；1 位 +10 护盾
- 道具：flip（反转数字）/edit（修改某位 ±1）/reroll（重抽）
- 自瞄（攻击自己）：未命中不扣血，命中扣自己血
- 杀光其它玩家者获胜

## 文件结构

```
ai_game/
├── scripts/                      # 训练脚本
│   ├── game.py                   # 游戏引擎（Python 移植）
│   ├── model.py                  # 神经网络模型
│   ├── numpy_net.py              # numpy 推理加速（2.6x）
│   ├── train.py                  # 单种群训练（REINFORCE）
│   ├── genetic_train.py          # 遗传算法 + 多进程训练
│   ├── visualize_train.py        # 可视化训练（生成 PNG 图表）
│   ├── play.py                   # 人机对弈 / 观战
│   ├── watch_compact.py          # 紧凑观战脚本
│   └── train_bounded.py          # 分批训练包装
├── download/ai_game/             # 训练成果
│   ├── model.pt                  # 当前最佳模型
│   ├── genetic_model.pt          # 遗传模型快照
│   ├── classic.txt               # 经典对局记录
│   ├── AI_TRAINING_SUMMARY.md    # 训练总结报告
│   ├── battle_record_v*.txt      # 各版本对战记录
│   └── viz/                      # 可视化图表
│       ├── win_rates_*.png       # 胜率曲线
│       ├── losses_*.png          # Loss 曲线
│       ├── best_win_rate_*.png   # 历史最佳
│       └── item_usage_*.png      # 道具使用统计
└── README.md                     # 本文件
```

## 环境要求

```
Python 3.10+
PyTorch 2.0+ (CPU 版即可)
NumPy
Matplotlib
```

## 安装

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install torch numpy matplotlib --index-url https://download.pytorch.org/whl/cpu
```

## 使用方法

### 1. 单种群训练

```bash
cd scripts
python3 train.py --iters 200 --games 48 --players 4
```

参数：
- `--iters`: 训练轮数
- `--games`: 每轮自博弈局数
- `--players`: 玩家数（2-10）
- `--lr`: 学习率
- `--temperature`: 采样温度

### 2. 遗传算法训练（推荐）

```bash
cd scripts
python3 genetic_train.py --gens 30 --iters 8 --workers 4
```

参数：
- `--gens`: 进化代数
- `--iters`: 每代训练轮数
- `--workers`: 并行进程数（建议 = CPU 核心数）
- `--seed`: 随机种子

4 个种群配置：
- A_relu_hot: ReLU 激活，高温 1.8，激进型
- B_gelu_balanced: GELU 激活，中温 1.4，平衡型
- C_tanh_conservative: Tanh 激活，低温 1.2，保守型
- D_elu_aggressive: ELU 激活，超高温 2.0，超激进型

### 3. 可视化训练

```bash
cd scripts
python3 visualize_train.py --gens 20 --iters 8 --workers 4 --viz-every 1
```

生成图表（保存到 `download/ai_game/viz/`）：
- `win_rates_*.png`: 4 个种群的胜率曲线
- `losses_*.png`: 训练 Loss 曲线
- `best_win_rate_*.png`: 历史最佳胜率
- `item_usage_*.png`: 道具使用统计

### 4. 观战 / 人机对弈

```bash
cd scripts
# AI 自对战
python3 watch_compact.py 1 0 0.5 80
# 参数: 局数 种子 温度 最大回合数

# 人机对弈
python3 play.py human --seat 0 --temp 0.1
# 参数: --seat 你的座位（0-3）, --temp AI 温度
```

### 5. 加载模型

模型路径：`download/ai_game/model.pt`
所有脚本自动加载最新最佳模型。

## 训练成果

当前最佳模型：遗传算法第 444 代
- 最佳种群：C_tanh_conservative
- 历史最佳胜率：61.1%
- 总训练：约 14000+ 局自博弈

## 技术架构

### 1. 游戏引擎（game.py）
- Python 版忠实移植自 C++ 代码
- 状态机：CHOOSE_TARGET → ITEM_DECISION → ITEM_CHOICE →
          EDIT_POS → EDIT_DELTA → REWARD_CHOICE → GAME_OVER
- 支持自瞄（攻击自己），自瞄命中扣自己血
- 121 维状态编码
- 道具使用追踪

### 2. 神经网络（model.py）
- 共享 MLP 主干：input(121) → 128 → 128 → 128
- 6 个动作头 + 价值头
- 总参数量：52252
- 支持多种激活函数（ReLU/GELU/Tanh/ELU）

### 3. 训练算法（train.py）
- REINFORCE 带价值基线
- 复合奖励 shaping（7 个维度）
- 自博弈数据收集 + 批量训练
- 支持续训

### 4. 遗传算法（genetic_train.py）
- 4 种群并行进化
- 多进程训练（ProcessPoolExecutor，绕过 GIL）
- 锦标赛评估（1v1 + 4 种群混战 + 多温度）
- 遗传操作：精英保留 + 线性交叉 + 高斯变异

### 5. NumpyNet 加速（numpy_net.py）
- 纯 numpy 推理，比 torch forward 快 2.6x
- 纯 numpy action_mask，比 torch 版快 17x
- 结果与 torch 完全一致（误差 1e-8）

## 奖励函数

```
reward = 0.05  × ΔHP_self           自己 HP 变化
       + 0.03  × (-ΔHP_opp)         对手 HP 减少
       + 0.05  × Δitems             道具净增量
       + 0.50  × hit_via_item       道具导致的额外命中
       + 0.30  × dmg_avoided        道具导致的伤害避免
       + (-0.03) × useless_via_item 无效道具使用扣分
       + 0.40  × self_aim_hurt      反向防御奖励
```

## AI 学到的策略

1. **自瞄策略**：残血时自瞄保命，规避 5/6 位+未命中的自扣血风险
2. **道具策略**：
   - flip：反向防御（让自瞄者反伤）、破坏对手命中
   - edit：精准调整数字位创造命中
   - reroll：攻击者求变
3. **道具组合**：reroll + edit×N、edit×N 连续修改
4. **位数策略**：1 血时追求 4 位数（规避自扣血 + 获得道具）
5. **集火策略**：优先攻击 HP 最低的玩家

## 训练历程

| 版本 | 训练量 | 胜率 | 道具使用 | 关键改动 |
|------|--------|------|----------|----------|
| v2   | 1480 轮 | -    | 0        | 基础训练 |
| v3   | 1965 轮 | -    | 745      | 加道具奖励 |
| v4   | 2545 轮 | -    | 62       | 加无效惩罚 |
| v5   | 2795 轮 | -    | 38       | 加反向防御 |
| v6   | 4085 轮 | -    | 237      | 恢复鼓励 |
| v8   | gen10  | 33.3% | 93      | 遗传算法引入 |
| v9   | gen27  | 38.9% | 176     | 多进程优化 |
| v10  | gen110 | 55.6% | 552     | 持续进化 |
| v11  | gen150 | 61.1% | 529     | 小惩罚回归 |
| 当前 | gen444 | 61.1% | ~500    | 高温 + 增强对抗 |

## 故障排查

### 1. torch 安装失败
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 中文字体显示问题
matplotlib 会自动 fallback 到 DejaVu Sans，不影响功能。

### 3. 多进程报错
确保 `--workers` 不超过 CPU 核心数。

### 4. 续训失败
删除 `download/ai_game/genetic_model.pt` 重新开始。

## 联系方式

如有问题，查看 `AI_TRAINING_SUMMARY.md` 获取完整训练报告。
