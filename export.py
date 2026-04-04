import torch
import yaml
import time
from pathlib import Path

# 导出 ONNX 必需的库
try:
    import onnx
    from onnxsim import simplify
except ImportError:
    print("❌ 错误: 未安装 onnx 或 onnxsim。请运行: uv pip install onnx onnxsim")
    exit(1)

# 可选：用于量化的库
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False

# 导入你的网络模型
from src.model import ArmorNet

def deploy_model(config_path="config.yaml"):
    # 1. 检查并读取配置
    if not Path(config_path).exists():
        print(f"❌ 找不到配置文件 {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config['model_info']['name']
    num_classes = config['model_info']['num_classes']
    input_w, input_h = config['model_info']['input_size'] # [宽, 高]
    
    # 定义路径
    deploy_dir = Path("deploy")
    pth_path = deploy_dir / f"{model_name}_best.pth"
    onnx_path = deploy_dir / f"{model_name}.onnx"
    sim_onnx_path = deploy_dir / f"{model_name}_sim.onnx"
    quant_onnx_path = deploy_dir / f"{model_name}_quant.onnx"

    if not pth_path.exists():
        print(f"❌ 找不到权重文件 {pth_path}，请先完成训练。")
        return

    print(f"🚀 开始部署流程: {model_name}")
    
    # 2. 初始化模型并加载权重
    device = torch.device("cpu")
    model = ArmorNet(num_classes=num_classes)
    
    # 使用 weights_only=True 以符合最新的安全实践
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval() # 必须切换到评估模式

    # 3. 执行 ONNX 导出
    # 注意：输入形状为 [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, 1, input_h, input_w, device=device)
    
    print("⏳ 正在导出原始 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},    
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ 原始 ONNX 已生成: {onnx_path}")

    # 4. 模型简化 (onnxsim)
    print("🛠️ 正在简化计算图...")
    onnx_model = onnx.load(str(onnx_path))
    model_simp, check = simplify(onnx_model)
    if check:
        onnx.save(model_simp, str(sim_onnx_path))
        print(f"✅ 简化版 ONNX 已生成: {sim_onnx_path}")
    else:
        print("⚠️ 模型简化失败，将跳过此步。")
        sim_onnx_path = onnx_path

    # 5. 动态量化 (INT8 压缩)
    if HAS_QUANT:
        print("🗜️ 正在执行 INT8 动态量化...")
        quantize_dynamic(
            model_input=str(sim_onnx_path),
            model_output=str(quant_onnx_path),
            weight_type=QuantType.QUInt8
        )
        print(f"✅ 量化版 ONNX 已生成: {quant_onnx_path}")
        
        # 打印文件大小对比
        orig_size = Path(onnx_path).stat().st_size / 1024
        quant_size = Path(quant_onnx_path).stat().st_size / 1024
        print(f"📊 体积压缩对比: {orig_size:.2f} KB -> {quant_size:.2f} KB (约 {orig_size/quant_size:.1f}x)")
    else:
        print("💡 提示: 未安装 onnxruntime，已跳过量化步骤。")

    print(f"\n🎉 部署准备完成！建议 C++ 端优先尝试使用: {sim_onnx_path.name}")

if __name__ == "__main__":
    deploy_model()