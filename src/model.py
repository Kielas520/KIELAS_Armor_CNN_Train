import torch
import torch.nn as nn

class ArmorNet(nn.Module):
    """
    RM 装甲板二值化特征分类网络
    Input: (B, 1, 28, 20)
    """
    def __init__(self, num_classes=4):
        super(ArmorNet, self).__init__()

        # ==========================================
        # Backbone: 特征提取层
        # ==========================================
        self.backbone = nn.Sequential(
            # Layer 1: 几何探测器 (1 -> 16)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 建议加入最大池化来降低特征图分辨率，减少显存占用并扩大感受野
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 16 x 14 x 10

            # Layer 2: 局部形状探测 (16 -> 32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 3: 拓扑联系强化 (32 -> 32)
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 输出: 32 x 7 x 5

            # Layer 4: 抽象语义特征 (32 -> 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 5: 高级特征 (64 -> 128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # Head: 分类层
        # ==========================================
        self.head = nn.Sequential(
            # 全空间聚合: 无论特征图多大，强行压缩为 1x1
            nn.AdaptiveAvgPool2d((1, 1)), # 输出: 128 x 1 x 1
            
            # 展平维度
            nn.Flatten(), # 输出: 128
            
            # FC1: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            # FC2: 64 -> num_classes (输出原始 Logits)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ================= 单独执行测试 =================
if __name__ == "__main__":
    import yaml
    from pathlib import Path
    
    print("🚀 开始独立测试 Model 模块...\n")
    
    # 尝试读取 config 获取类别数和输入尺寸
    num_classes = 4
    input_h, input_w = 28, 20
    
    if Path("config.yaml").exists():
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            num_classes = cfg['model_info']['num_classes']
            input_w = cfg['model_info']['input_size'][0]
            input_h = cfg['model_info']['input_size'][1]
    
    # 初始化模型
    model = ArmorNet(num_classes=num_classes)
    
    # 创建一个模拟的输入张量: [BatchSize, Channels, Height, Width]
    dummy_input = torch.randn(2, 1, input_h, input_w)
    
    # 前向传播测试
    outputs = model(dummy_input)
    
    print(f"📦 输入张量形状: {dummy_input.shape}")
    print(f"✅ 输出张量形状: {outputs.shape} (预期: [2, {num_classes}])")
    
    # 打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型总参数量: {total_params:,}")
    
    # 验证 AdaptiveAvgPool2d 的抗畸变能力
    # 即使输入尺寸变成 40x40，全连接层依然不会报错
    test_robust_input = torch.randn(2, 1, 40, 40)
    robust_output = model(test_robust_input)
    print(f"🛡️ 变长输入测试 (40x40) 输出形状: {robust_output.shape} -> 通过！")