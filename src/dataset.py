import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

class ArmorDataset(Dataset):
    def __init__(self, config, mode='processed', transform=None):
        self.config = config
        self.mode = mode
        self.transform = transform
        self.input_w = config['model_info']['input_size'][0]
        self.input_h = config['model_info']['input_size'][1]
        
        self.raw_dir = Path("data/raw")
        self.proc_dir = Path("data/processed")
        self.label_map = {'1': 0, '2': 1, '3': 2, '4negative': 3}

        if mode == 'processed':
            csv_path = self.proc_dir / "labels.csv"
            if not csv_path.exists():
                print("⚠️ 自动触发预处理...")
                ArmorDataset.preprocess(config)
            self.data_info = pd.read_csv(csv_path)
        else:
            self.data_info = self._scan_raw()

    def _scan_raw(self):
        temp_list = []
        for folder, idx in self.label_map.items():
            paths = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
                paths.extend(list((self.raw_dir / folder).glob(ext)))
            for p in paths:
                temp_list.append({'path': str(p), 'class_id': idx})
        return pd.DataFrame(temp_list)

    @staticmethod
    def preprocess(config):
        """执行离线预处理：数据平衡 + 强制尺寸校验"""
        input_w = config['model_info']['input_size'][0]
        input_h = config['model_info']['input_size'][1]
        
        raw_dir = Path("data/raw")
        proc_dir = Path("data/processed")
        label_map = {'1': 0, '2': 1, '3': 2, '4negative': 3}
        
        paths_by_class = {idx: [] for idx in range(4)}
        for folder, idx in label_map.items():
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
                paths_by_class[idx].extend(list((raw_dir / folder).glob(ext)))
        
        counts = {k: len(v) for k, v in paths_by_class.items()}
        print(f"📊 Raw Stats: {counts}")
        min_samples = min(counts.values())
        print(f"⚖️ Balancing: Each class will have {min_samples} samples")

        processed_records = []
        project_root = Path.cwd().absolute()

        for class_id, paths in paths_by_class.items():
            np.random.shuffle(paths)
            selected = paths[:min_samples]
            save_dir = proc_dir / str(class_id)
            save_dir.mkdir(parents=True, exist_ok=True)

            for i, p in enumerate(tqdm(selected, desc=f"Processing Class {class_id}")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                if img.shape[1] != input_w or img.shape[0] != input_h:
                    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)
                
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                new_name = f"{class_id}_{i:05d}.jpg"
                save_path = (save_dir / new_name).absolute()
                cv2.imwrite(str(save_path), img)
                
                processed_records.append({
                    'path': str(save_path.relative_to(project_root)),
                    'class_id': class_id,
                    'weight': 1.0
                })
        
        pd.DataFrame(processed_records).to_csv(proc_dir / "labels.csv", index=False)
        print(f"✅ Preprocessing Done. Labels saved to {proc_dir / 'labels.csv'}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img = cv2.imread(str(row['path']), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return torch.zeros((1, self.input_h, self.input_w)), int(row['class_id'])

        if img.shape[1] != self.input_w or img.shape[0] != self.input_h:
            img = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        
        return img, int(row['class_id'])

# ================= 新增/修复的部分 =================
def get_dataloader(config, mode='processed', is_train=True):
    """
    统一的数据加载入口，动态分配 transform 策略
    """
    aug_cfg = config['augmentation']
    input_size = (config['model_info']['input_size'][1], config['model_info']['input_size'][0]) # (H, W)

    if is_train:
        # 训练时（或者预览增强效果时），开启所有数据增强
        data_transforms = transforms.Compose([
            transforms.ToPILImage(), # 将 cv2 读取的 numpy 数组转为 PIL 图像以供后续增强
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
    else:
        # 验证/测试时，仅转换为 Tensor 并二值化
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > aug_cfg['binary_threshold']).float())
        ])

    dataset = ArmorDataset(config, mode=mode, transform=data_transforms)
    
    return DataLoader(
        dataset, 
        batch_size=config['model_info']['batch_size'], 
        shuffle=is_train,
        num_workers=0
    )

# ================= 单独执行测试 =================
if __name__ == "__main__":
    print("🚀 开始独立测试 Dataset 模块...\n")
    
    # 1. 测试配置加载
    config_file = "config.yaml"
    if not Path(config_file).exists():
        print(f"❌ 找不到 {config_file}，请确保在项目根目录下运行。")
        exit(1)
        
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print("✅ 配置文件读取成功！")
    
    # 2. 测试预处理功能 (如果 processed 没东西，这里会自动触发)
    print("\n--- 测试 Processed 模式 ---")
    try:
        # 修改测试用的 batch_size 避免一次加载太多
        cfg['model_info']['batch_size'] = 4 
        proc_loader = get_dataloader(cfg, mode='processed', is_train=False)
        proc_imgs, proc_lbls = next(iter(proc_loader))
        print(f"✅ Processed Dataloader 初始化成功！")
        print(f"📦 Batch Shape: {proc_imgs.shape} (预期: [4, 1, {cfg['model_info']['input_size'][1]}, {cfg['model_info']['input_size'][0]}])")
        print(f"🏷️ Labels: {proc_lbls}")
    except Exception as e:
        print(f"❌ Processed 模式测试失败: {e}")

    # 3. 测试 Raw 模式 (包含增强逻辑)
    print("\n--- 测试 Raw 模式 (带数据增强) ---")
    try:
        raw_loader = get_dataloader(cfg, mode='raw', is_train=True)
        raw_imgs, raw_lbls = next(iter(raw_loader))
        print(f"✅ Raw Dataloader 初始化成功！")
        print(f"📦 Batch Shape: {raw_imgs.shape}")
        print(f"🏷️ Labels: {raw_lbls}")
    except Exception as e:
        print(f"❌ Raw 模式测试失败: {e}")
        
    print("\n🎉 测试结束。如果以上全部打印出 Batch Shape 且没有报错，说明数据流已彻底打通！")