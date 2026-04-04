import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter  # 新增：导入 TensorBoard

# 导入自定义模块
from src.dataset import ArmorDataset
from src.model import ArmorNet

# ================= 数据包装器 =================
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# ================= 辅助函数 =================
def get_transforms(config):
    aug_cfg = config['augmentation']
    input_size = (config['model_info']['input_size'][1], config['model_info']['input_size'][0])
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=aug_cfg['rotation_degree']),
        transforms.RandomResizedCrop(
            size=input_size, 
            scale=(aug_cfg['scale_min'], aug_cfg['scale_max']), 
            ratio=(aug_cfg['ratio_min'], aug_cfg['ratio_max'])
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=aug_cfg['erasing_p'], 
            scale=tuple(aug_cfg['erasing_scale']), 
            value=0
        ),
        transforms.Lambda(lambda x: (x > aug_cfg['binary_threshold']).float())
    ])
    
    eval_transform = transforms.Compose([
        transforms.Lambda(lambda x: (x > aug_cfg['binary_threshold']).float())
    ])
    
    return train_transform, eval_transform

# ================= 主训练循环 =================
def train_model(config_path="config.yaml"):
    if not Path(config_path).exists():
        print(f"❌ 找不到配置文件 {config_path}")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config['model_info']['name']
    num_classes = config['model_info']['num_classes']
    learning_rate = config['model_info']['learning_rate']
    epochs = config['model_info']['epochs']
    batch_size = config['model_info']['batch_size']
    class_names = [config['class_names'][i] for i in range(num_classes)]
    
    cfg_device = config['model_info'].get('device', 'auto').lower()
    if cfg_device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
        
    print(f"🚀 开始训练 [{model_name}] | 使用设备: {device}")
    
    # 初始化 TensorBoard Writer
    log_dir = Path("runs") / model_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"📈 TensorBoard 日志将保存在: {log_dir}")
    
    train_transform, eval_transform = get_transforms(config)
    full_dataset_raw = ArmorDataset(config, mode='processed', transform=None)
    total_samples = len(full_dataset_raw)
    
    split_cfg = config.get('dataset_split', {})
    use_val = split_cfg.get('use_validation', True)
    use_test = split_cfg.get('use_test', True)
    ratios = split_cfg.get('ratios', [0.8, 0.1, 0.1])
    
    val_len = int(total_samples * ratios[1]) if use_val else 0
    test_len = int(total_samples * ratios[2]) if use_test else 0
    train_len = total_samples - val_len - test_len 
    
    splits = random_split(full_dataset_raw, [train_len, val_len, test_len])
    
    train_dataset = DatasetWrapper(splits[0], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = None
    if use_val and val_len > 0:
        val_dataset = DatasetWrapper(splits[1], transform=eval_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_loader = None
    if use_test and test_len > 0:
        test_dataset = DatasetWrapper(splits[2], transform=eval_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ArmorNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    save_dir = Path("deploy")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        # ---------- 训练阶段 ----------
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 写入 TensorBoard：记录训练集 Loss 和当前学习率
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # ---------- 验证与评估阶段 ----------
        if val_loader:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            current_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            
            # 写入 TensorBoard：记录验证集 Loss 和 Accuracy
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Val', current_acc * 100, epoch)

            scheduler.step(avg_val_loss)

            print(f"\n📊 Epoch {epoch+1} 总结:")
            print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {current_acc*100:.2f}%")
            
            if (epoch + 1) % 5 == 0 or current_acc > best_val_acc:
                print("   --- 分类评估报告 ---")
                print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
                print("   --- 混淆矩阵 (行:真实值, 列:预测值) ---")
                print(confusion_matrix(all_labels, all_preds))
                print("-" * 50)

            if current_acc > best_val_acc:
                best_val_acc = current_acc
                save_path = save_dir / f"{model_name}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"💾 验证集准确率提升至 {best_val_acc*100:.2f}%，权重已保存。\n")
                
            if best_val_acc >= 0.995 and avg_val_loss < 0.03:
                print(f"🎯 达到停止条件 (Acc >= 99.5%, Loss < 0.03)，提前结束训练。")
                break
        else:
            save_path = save_dir / f"{model_name}_last.pth"
            torch.save(model.state_dict(), save_path)
            print(f"\n📊 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

    # ================= 最终测试集评估 =================
    if test_loader:
        print("\n" + "="*50)
        print("🚀 开始最终测试集 (Test Set) 评估...")
        
        best_model_path = save_dir / f"{model_name}_best.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            print("✅ 已加载最佳模型权重。")
            
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
        test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
        print(f"\n🏆 最终测试集准确率: {test_acc*100:.2f}%")
        print("   --- 测试集分类报告 ---")
        print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))
        print("   --- 测试集混淆矩阵 ---")
        print(confusion_matrix(test_labels, test_preds))
        
        # 将最终测试准确率也写入 TensorBoard
        writer.add_text('Test/Accuracy', f"{test_acc*100:.2f}%", 0)

    writer.close()
    print(f"\n🎉 炼丹结束！")
    print(f"👉 查看曲线：在终端运行 `tensorboard --logdir=runs`")

if __name__ == "__main__":
    train_model()