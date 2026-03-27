## 代码架构
```text
├── configs/                # 配置文件
├── dataset/                # 数据流相关
│   ├── __init__.py
│   ├── datasets.py         # 自定义 Dataset (如图像分类、分割数据集)
│   ├── transforms.py       # 数据增强 (基于 albumentations 或 torchvision)
│   └── dataloaders.py      # DataLoader 封装与构建工厂
├── models/                 # 模型定义
├── layers/                 # 基础组件
├── loss/                   # 损失函数
│   ├── __init__.py         # 带有自动注册机制 (Registry)
│   ├── loss1.py            
│   └── loss2.py       
├── optim/                  # 优化器构建工厂 (封装 SGD, AdamW 等)
├── scheduler/              # 学习率调度器构建工厂 (CosineAnnealing 等)
├── engine/                 # 核心指挥部！
│   ├── __init__.py
│   ├── train_engine.py     # 训练器基类 (包含 train, train_one_epoch, val, save)
│   └── metrics.py          # 评估指标 (Accuracy, IoU 等)
├── scripts/                # 工具脚本 (已完成自动生成 __init__ 🏆)
├── tools/                  # 暴露给用户的执行入口！
│   ├── train.py            # 训练主程序 (入口)
│   ├── test.py             # 评估主程序
│   └── infer.py            # 单张/批量图片推理脚本
└── test/                   # 单元测试代码
```
