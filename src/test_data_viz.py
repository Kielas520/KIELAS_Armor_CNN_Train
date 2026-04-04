import yaml
import matplotlib.pyplot as plt
from src.dataset import get_dataloader # 仅通过接口获取数据

def test_viz(mode='processed', show_augmentation=False):
    """
    mode: 'raw' 或 'processed'
    show_augmentation: 是否开启 config 中的数据增强进行预览
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 修改预览专用的 batch_size
    config['model_info']['batch_size'] = 8
    
    # 严格调用集中式 loader
    loader = get_dataloader(config, mode=mode, is_train=show_augmentation)
    
    try:
        images, labels = next(iter(loader))
    except StopIteration:
        print(f"❌ 目录 data/{mode} 下未找到数据。")
        return
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"Batch Shape: {images.shape} (Expected: [8, 1, 28, 20])")
    
    plt.figure(figsize=(15, 5))
    title = f"Mode: {mode.upper()} | Augmentation: {'ON' if show_augmentation else 'OFF'}"
    plt.suptitle(title, fontsize=14)
    
    for i in range(len(images)):
        plt.subplot(1, 8, i+1)
        # 显式指定 vmin 和 vmax 保证二值图显示不失真
        plt.imshow(images[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 如果 data/processed 为空，取消下面两行的注释跑一次即可
    # import yaml; from src.dataset import ArmorDataset
    # with open("config.yaml", "r") as f: ArmorDataset.preprocess(yaml.safe_load(f))
    
    # 建议测试：查看 processed 目录下的图，并开启增强看看训练时它会被扭曲成什么样
    test_viz(mode='raw', show_augmentation=True)
    test_viz(mode='processed', show_augmentation=True)