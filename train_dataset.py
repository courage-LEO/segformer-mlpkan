import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from albumentations.pytorch import ToTensorV2
import albumentations as A
from MLPKANSegFormer import MLPKANSegFormer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


# 数据集定义
class CustomDataset(Dataset):
    def __init__(self, data_root_dir, mode="train", num_classes=20):
        self.mode = mode
        self.num_classes = num_classes
        self.transform = A.Compose([
            A.Resize(64, 64),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0)
            ], p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.image_paths, self.label_paths = [], []
        seq_dict = {"train": list(range(1, 16)) + list(range(31, 36)),
                    "val": [16, 17, 18, 19, 20, 36, 37],
                    "test": list(range(21, 30)) + list(range(38, 42))}

        for seq_id in seq_dict.get(mode, []):
            seq_dir = os.path.join(data_root_dir, f"uavid_{mode}", f"seq{seq_id}")
            img_dir, label_dir = os.path.join(seq_dir, "images"), os.path.join(seq_dir, "labels")
            self.image_paths.extend(
                sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]))
            if mode != "test":
                self.label_paths.extend(
                    sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")]))

        if mode != "test":
            assert len(self.image_paths) == len(self.label_paths), "图像与标签数量不匹配!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img_rgb)
        img_t = transformed["image"]

        if self.mode == "test":
            return img_t

        label_img = np.array(cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE))
        transformed = self.transform(image=img_rgb, mask=label_img)
        label_t = transformed["mask"].long().clamp(0, self.num_classes - 1)
        return img_t, label_t


# 计算 mIoU
def calculate_miou(preds, labels, num_classes=20):
    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
    iou_per_class = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum()
        union = ((preds == cls) | (labels == cls)).sum()
        iou_per_class.append(intersection / union if union > 0 else float('nan'))
    return np.nanmean(iou_per_class)


# 训练主函数
def main():
    root_dir = r"C:\Users\fakedd\Downloads\uavid_v1.5_official_release_image\uavid_v1.5_official_release_image"

    # 数据加载
    batch_size = 64
    train_dataset, val_dataset = CustomDataset(root_dir, "train"), CustomDataset(root_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 模型初始化
    model = MLPKANSegFormer(input_height=64, class_num=20).to(device)

    # 训练参数
    epochs = 2000
    lr = 3e-4
    weight_decay = 2e-4

    # 优化器 & 调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)  # 余弦退火学习率
    lr_scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30, verbose=True)

    # ✅ Label Smoothing 以减少过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 指数移动平均（EMA）
    ema_decay = 0.99
    ema_model = MLPKANSegFormer(input_height=64, class_num=20).to(device)
    ema_model.load_state_dict(model.state_dict())
    best_miou = 0.0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss, start_time = 0.0, time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss_seg = criterion(logits, labels)

            # KAN 正则化
            loss_reg = 0.0
            if hasattr(model.Dec, 'kan_layer'):
                loss_reg = model.Dec.kan_layer.regularization_loss(0.1, 0.1)
            loss = loss_seg + 0.05 * loss_reg  # 提高 loss_reg 的影响力

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss_seg.item()

            # EMA 更新
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        train_loss /= len(train_loader)
        scheduler.step()

        # 评估
        model.eval()
        val_loss, total_miou = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_logits = ema_model(images)  # 用 EMA 模型评估
                loss_seg = criterion(val_logits, labels)
                val_loss += loss_seg.item()
                total_miou += calculate_miou(torch.argmax(val_logits, dim=1), labels)

        val_loss /= len(val_loader)
        avg_miou = total_miou / len(val_loader)
        lr_scheduler_plateau.step(avg_miou)

        if avg_miou > best_miou:
            best_miou = avg_miou
            torch.save(model.state_dict(), "best_model.pth")


        print(
            f"[Epoch {epoch + 1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {avg_miou:.4f} | Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), "segformer_uavid.pth")


if __name__ == "__main__":
    main()