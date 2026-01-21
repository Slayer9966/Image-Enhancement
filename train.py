

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import kornia
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler

# ----------------------------
# Parameters & Configuration
# ----------------------------
num_curve_points = 16 
GRID_SIZE = 16        # 16x16 local zones
GRID_DEPTH = 8        # 8 intensity bins (Shadows to Highlights)
TARGET_SIZE = 1024    

# ----------------------------
# Bilateral Slicing & Curve Application
# ----------------------------
def apply_hybrid_curves(img, global_params, grid_params):
    """
    img: [B, 3, H, W]
    global_params: [B, 3, 16] -> One curve for the whole image
    grid_params: [B, 3 * 16 * 8, 16, 16] -> Local curves
    """
    b, c, h, w = img.shape

    # --- PART 1: APPLY GLOBAL CURVE ---
    # Global curve serves as a stable "base" for the whole image
    global_curves = torch.cumsum(torch.softmax(global_params, dim=2), dim=2)
    
    # Simple global application
    img_flat = img.view(b, c, -1)
    indexed_g = img_flat * (num_curve_points - 1)
    idx_l_g = indexed_g.long().clamp(0, num_curve_points - 2)
    idx_h_g = idx_l_g + 1
    v_l_g = torch.gather(global_curves, 2, idx_l_g)
    v_h_g = torch.gather(global_curves, 2, idx_h_g)
    dist_g = indexed_g - idx_l_g.float()
    img_global = (v_l_g + dist_g * (v_h_g - v_l_g)).view(b, c, h, w)

    # --- PART 2: APPLY LOCAL BILATERAL GRID ---
    # The grid now only has to fix what the global curve missed (Residual)
    grid = grid_params.view(b, 3 * num_curve_points, GRID_DEPTH, GRID_SIZE, GRID_SIZE)
    guide = (0.299 * img_global[:, 0:1] + 0.587 * img_global[:, 1:2] + 0.114 * img_global[:, 2:3] * 2.0 - 1.0).clamp(-1, 1)
    
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).to(img.device).unsqueeze(0).expand(b, h, w, 2)
    full_coords = torch.cat([coords, guide.permute(0, 2, 3, 1)], dim=-1).unsqueeze(1)

    sliced_params = F.grid_sample(grid, full_coords, align_corners=True, mode='bilinear').squeeze(2)
    sliced_params = sliced_params.view(b, 3, num_curve_points, h, w)
    local_curves = torch.cumsum(torch.softmax(sliced_params, dim=2), dim=2)

    out_channels = []
    for i in range(3):
        img_c = img_global[:, i:i+1] # We apply local curves to the ALREADY globally corrected image
        curve_c = local_curves[:, i]
        indexed = img_c * (num_curve_points - 1)
        idx_l = indexed.long().clamp(0, num_curve_points - 2)
        idx_h = idx_l + 1
        v_l = torch.gather(curve_c, 1, idx_l)
        v_h = torch.gather(curve_c, 1, idx_h)
        dist = indexed - idx_l.float()
        out_channels.append(v_l + dist * (v_h - v_l))

    return torch.cat(out_channels, dim=1).clamp(0, 1)

# ----------------------------
# Dataset Class
# ----------------------------
class ImageToParamDataset(Dataset):
    def __init__(self, images_dir, gt_dir, augment=False):
        self.images_dir, self.gt_dir = images_dir, gt_dir
        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))])
        self.augment = augment

    def __len__(self): return len(self.images)

    def pad_to_square(self, img, target_size=TARGET_SIZE):
        h, w = img.shape[:2]
        pad_h = max(0, target_size - h)
        pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
        pad_w = max(0, target_size - w)
        pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        gt_path = os.path.join(self.gt_dir, self.images[idx])
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = TARGET_SIZE / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if self.augment and np.random.random() > 0.5:
            img, gt = cv2.flip(img, 1), cv2.flip(gt, 1)

        img = self.pad_to_square(img)
        gt = self.pad_to_square(gt)
        
        img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        gt_t = torch.from_numpy(gt.transpose(2,0,1)).float() / 255.0
        return img_t, gt_t

# ----------------------------
# Bilateral Grid Model
# ----------------------------
class HybridEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = smp.encoders.get_encoder("resnet18", in_channels=3, weights="imagenet")
        
        # Branch 1: Global Curve (The "Holistic" look)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3 * num_curve_points)
        )

        # Branch 2: Bilateral Grid (The "Local" fix)
        self.grid_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3 * num_curve_points * GRID_DEPTH, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        feat = self.backbone(x)[-1] 
        
        # Predict both
        global_params = self.global_head(feat).view(-1, 3, num_curve_points)
        
        feat_grid = F.interpolate(feat, size=(GRID_SIZE, GRID_SIZE), mode='bilinear', align_corners=True)
        grid_params = self.grid_head(feat_grid)
        
        return global_params, grid_params

# ----------------------------
# Loss & Training
# ----------------------------
def compute_loss(global_p, grid_p, input_img, gt_img, ssim_module):
    enhanced = apply_hybrid_curves(input_img, global_p, grid_p)
    
    l1_loss = F.l1_loss(enhanced, gt_img)
    ssim_loss = ssim_module(enhanced, gt_img)
    
    # TV Regularization for the grid
    grid_res = grid_p.view(-1, 3 * num_curve_points, GRID_DEPTH, GRID_SIZE, GRID_SIZE)
    tv_s = torch.mean(torch.abs(grid_res[:, :, :, :, 1:] - grid_res[:, :, :, :, :-1])) + \
           torch.mean(torch.abs(grid_res[:, :, :, 1:, :] - grid_res[:, :, :, :-1, :]))
    tv_d = torch.mean(torch.abs(grid_res[:, :, 1:, :, :] - grid_res[:, :, :-1, :, :]))

    total_loss = l1_loss + 0.5 * ssim_loss + (0.01 * tv_s) + (0.005 * tv_d)
    return total_loss, l1_loss, enhanced

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training Bilateral Grid (Local Correction) on: {device}")
    
    model_save_path = "models"
    os.makedirs(model_save_path, exist_ok=True)

    full_dataset = ImageToParamDataset("DataSet_Resized/Images", "DataSet_Resized/GT", augment=True)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=10)

    model = HybridEnhancer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    ssim_module = kornia.losses.SSIMLoss(11).to(device)
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')

    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                global_p, grid_p = model(img) # Unpack the tuple here
                loss, _, _ = compute_loss(global_p, grid_p, img, gt, ssim_module)
                        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img, gt in val_loader:
                img, gt = img.to(device), gt.to(device)
                with autocast(device_type='cuda'):
                    global_p, grid_p = model(img) # Unpack the tuple here
                    loss, _, _ = compute_loss(global_p, grid_p, img, gt, ssim_module)
                total_val_loss += loss.item()
        
        avg_val = total_val_loss / len(val_loader)
        scheduler.step(avg_val)
        print(f"Epoch [{epoch+1}/100] Train: {total_train_loss/len(train_loader):.4f} Val: {avg_val:.4f}")

        torch.save(model.state_dict(), os.path.join(model_save_path, "latest_bilateral_model.pth"))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_bilateral_model.pth"))
            print(f"⭐ New Best Local-Adjustment Model saved!")

if __name__ == "__main__":
    main()