
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from pytorch_msssim import SSIM
from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.low_images = sorted(os.listdir(low_dir))
        self.normal_images = sorted(os.listdir(normal_dir))
        assert len(self.low_images) == len(self.normal_images), "Количество изображений в 'low' и 'normal' должно быть одинаковым"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        normal_path = os.path.join(self.normal_dir, self.normal_images[idx])

        low_image = Image.open(low_path).convert('RGB')
        normal_image = Image.open(normal_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            normal_image = self.transform(normal_image)

        return low_image, normal_image

def tv_loss(tensor):
    batch_size = tensor.size(0)
    h = tensor.size(2)
    w = tensor.size(3)
    tv_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).sum()
    tv_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (batch_size * h * w)

def histogram_loss(L, bins=100):
    L = torch.clamp(L, 0, 1)
    L_flat = L.detach().cpu().flatten()
    hist, _ = torch.histogram(L_flat, bins=bins, range=(0, 1))
    hist = hist / (hist.sum() + 1e-6)  # Нормализуем
    hist = hist[hist > 0]  # Избегаем log(0)
    entropy = -torch.sum(hist * torch.log(hist))
    max_entropy = torch.log(torch.tensor(bins, dtype=torch.float32))
    return 1.0 - (entropy / max_entropy)  # Хотим максимизировать энтропию
def brightness_loss(L, target_mean=0.5, min_threshold=0.1, max_threshold=0.9):
    mean_L = torch.mean(L)
    var_L = torch.var(L)
    mean_loss = (mean_L - target_mean) ** 2
    var_loss = torch.relu(min_threshold - var_L)
    return mean_loss + 0.1 * var_loss
def brightness_constraint_R(R, target_mean=0.5):
    mean_R = torch.mean(R, dim=(2, 3), keepdim=True)
    return torch.mean((mean_R - target_mean) ** 2)
def tv_loss(R):
    grad_x = torch.abs(R[:, :, :, :-1] - R[:, :, :, 1:]).sum()
    grad_y = torch.abs(R[:, :, :-1, :] - R[:, :, 1:, :]).sum()
    return (grad_x + grad_y) / R.size(0)
def contrast_loss(L, bins=100):
    L = torch.clamp(L, 0, 1)
    var_L = torch.var(L)
    return torch.clamp(0.5 - var_L, min=0)
def variation_loss(R):
    R = torch.clamp(R, 0, 1)
    grad_x = torch.abs(R[:, :, :, :-1] - R[:, :, :, 1:]).mean()
    grad_y = torch.abs(R[:, :, :-1, :] - R[:, :, 1:, :]).mean()
    channel_var = torch.var(R, dim=1).mean()
    spatial_var = torch.var(R, dim=(2, 3)).mean()
    return 10.0 * (grad_x + grad_y) + channel_var + spatial_var
def l2_regularization(R):
    return torch.mean(R ** 2)
def min_L_loss(L, min_threshold=0.1):
    return torch.mean(torch.clamp(min_threshold - L, min=0))
def log_histogram(tensor, name, epoch, batch, log_file):
    tensor = torch.clamp(tensor, 0, 1)
    tensor_np = tensor.detach().cpu().flatten().numpy()
    hist, _ = np.histogram(tensor_np, bins=100, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-6)
    log_entry = f"{name} Histogram Epoch [{epoch}/{num_epochs_decom}], Batch [{batch+1}/{len(dataloader)}]: {hist.tolist()}\n"
    log_file.write(log_entry)
def channel_diversity_loss(R):
    R_mean_per_channel = torch.mean(R, dim=(0, 2, 3))
    mean_diff = torch.var(R_mean_per_channel)
    return torch.clamp(0.2 - mean_diff, min=0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOW_DIR = 'C:/NIR_LOL_C/dataset/train/low'
    NORMAL_DIR = 'C:/NIR_LOL_C/dataset/train/normal'
    transform = transforms.Compose([transforms.Resize((340, 512)), transforms.ToTensor()])
    dataset = LowLightDataset(low_dir=LOW_DIR, normal_dir=NORMAL_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16)
    accum_steps = 4

    VAL_LOW_DIR = 'C:/NIR_LOL_C/dataset/eval/low'
    VAL_NORMAL_DIR = 'C:/NIR_LOL_C/dataset/eval/normal'
    val_dataset = LowLightDataset(low_dir=VAL_LOW_DIR, normal_dir=VAL_NORMAL_DIR, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=16)

    decom_model = DecomNet().to(device)
    enhance_model = EnhanceNet().to(device)
    
    ssim_module_rgb = SSIM(data_range=1.0, size_average=True, channel=3).to(device)

    ssim_module_channel = SSIM(data_range=1.0, size_average=True, channel=1).to(device)

    criterion_mse = nn.MSELoss()

    recon_R_params = list(decom_model.recon_R.parameters())
    recon_R_optimizer = optim.Adam(decom_model.parameters(), lr=0.00001, weight_decay=1e-5)
    other_params = [p for n, p in decom_model.named_parameters() if "recon_R" not in n]
    other_optimizer = optim.Adam(other_params, lr=1e-5)

    scaler = torch.amp.GradScaler('cuda')

    for m in decom_model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    num_epochs_decom = 500
    patience = 20
    log_file = open("decom_training_log.txt", "w")
    

    print("Training Decom model...")
    best_loss_decom = float('inf')
    best_target_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs_decom):
        other_optimizer.zero_grad()
        recon_R_optimizer.zero_grad()
        total_target_loss = 0.0
        total_loss = 0.0
        total_recon = 0.0
        total_R = 0.0
        total_color = 0.0
        total_brightness = 0.0
        total_contrast = 0.0
        total_variation = 0.0
        total_l2 = 0.0
        total_channel_diversity = 0.0
        total_brightness_R = 0.0
        total_histogram_L = 0.0
        num_batches = 0

        loss_illumination_smooth = 0
        loss_reflectance_sim = 0
        decom_model.train()
        for i, (input_image, target_image) in enumerate(dataloader):
            input_image, target_image = input_image.to(device), target_image.to(device)

            R, L = decom_model(input_image)
            L = torch.clamp(L, 0, 1)
            L = torch.nan_to_num(L, nan=0.5, posinf=1.0, neginf=0.0)
            L_expanded = L.expand_as(R)

            R = torch.clamp(R, 0, 1)
            R = torch.nan_to_num(R, nan=0.5, posinf=1.0, neginf=0.0)
            target_image = torch.clamp(target_image, 0, 1)
            target_image = torch.nan_to_num(target_image, nan=0.5, posinf=1.0, neginf=0.0)
            log_histogram(L, "L", epoch, i, log_file)
            log_histogram(R, "R", epoch, i, log_file)
            L_mean = torch.mean(L).item()
            L_var = torch.var(L).item()
            R_mean = torch.mean(R).item()
            R_var = torch.var(R).item()
            log_entry = (f"Epoch [{epoch}/{num_epochs_decom}], Batch [{i+1}/{len(dataloader)}], "
                        f"L mean: {L_mean}, L var: {L_var}, "
                        f"R mean: {R_mean}, R var: {R_var}\n")
            log_file.write(log_entry)

            loss_recon = criterion_mse(R * L_expanded, input_image)
            loss_R = 1.0 - ssim_module_rgb(R, target_image)
            loss_color = sum(1.0 - ssim_module_channel(R[:, i:i+1, :, :], target_image[:, i:i+1, :, :]) for i in range(3))
            loss_brightness_L = brightness_loss(L, target_mean=0.5, min_threshold=0.1, max_threshold=0.9)
            loss_contrast_L = contrast_loss(L, bins=100)
            loss_variation_R = variation_loss(R)
            loss_l2_R = l2_regularization(R)
            loss_channel_diversity = channel_diversity_loss(R)
            loss_tv_R = tv_loss(R)
            loss_brightness_R = brightness_constraint_R(R, target_mean=0.5)
            loss_histogram_L = histogram_loss(L)
            loss_illumination_smooth = tv_loss(L)
            loss_reflectance_sim = criterion_mse(R, target_image)
            l2_reg_L = 0.0
            for param in decom_model.recon_L.parameters():
                l2_reg_L += torch.norm(param, p=2)
            
            l2_reg_R = 0.0
            for param in decom_model.recon_R.parameters():
                l2_reg_R += torch.norm(param, p=2)

            loss = (
                0.000001 * loss_illumination_smooth +
                5.0 * loss_recon +
                0.0 * loss_R +
                1.0 * loss_color +
                2.0 * loss_reflectance_sim +
                0 * loss_brightness_L +
                0 * loss_contrast_L +
                0 * loss_variation_R +
                0 * loss_l2_R +
                0 * loss_channel_diversity +
                0 * loss_tv_R +
                0 * loss_brightness_R +
                1.0 * loss_histogram_L +
                0 * l2_reg_L +
                0 * l2_reg_R
            )
            target_loss = (
                1.0 * loss_recon
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(decom_model.parameters(), max_norm=5.0)
            log_entry = f"Gradient norm: {grad_norm.item()}\n"
            log_file.write(log_entry)
            if grad_norm.item() < 1e-6:
                print(f"Warning: Gradient norm is too small: {grad_norm.item()}")

            # Логируем градиенты параметров
            for name, param in decom_model.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    log_entry = f"Gradient mean for {name}: {grad_mean}\n"
                    log_file.write(log_entry)

            param_before = {name: param.clone() for name, param in decom_model.named_parameters()}
            other_optimizer.step()
            recon_R_optimizer.step()
                
            other_optimizer.zero_grad()
            recon_R_optimizer.zero_grad()

            total_target_loss += target_loss.item()
            total_loss += loss.item()
            num_batches += 1

            R_mins = torch.amin(R, dim=[0, 2, 3]).tolist()
            R_maxs = torch.amax(R, dim=[0, 2, 3]).tolist()
            log_entry = (f"Epoch [{epoch}/{num_epochs_decom}], Batch [{i+1}/{len(dataloader)}], "
                        f"R [R,G,B] min: {R_mins}, max: {R_maxs}, "
                        f"L min: {L.min().item()}, max: {L.max().item()}, "
                        f"loss_recon: {loss_recon.item()}, loss_R: {loss_R.item()}, "
                        f"loss_color: {loss_color.item()}, loss_brightness_L: {loss_brightness_L.item()}, "
                        f"loss_contrast_L: {loss_contrast_L.item()}, "
                        f"loss_variation_R: {loss_variation_R.item()}, "
                        f"loss_l2_R: {loss_l2_R.item()}, "
                        f"loss_channel_diversity: {loss_channel_diversity.item()}, "
                        f"loss_tv_R: {loss_tv_R.item()}, "
                        f"loss_brightness_R: {loss_brightness_R.item()}, "
                        f"loss_histogram_L: {loss_histogram_L.item()}\n")
            log_file.write(log_entry)
        
        print(f"Decom Epoch [{epoch}/{num_epochs_decom}], Loss: {loss.item()}")
        vutils.save_image(R, f"decom_R_epoch_{epoch}.png", normalize=False)
        vutils.save_image(L, f"decom_L_epoch_{epoch}.png", normalize=False)

        avg_loss = total_loss / num_batches
        avg_target_loss = total_target_loss / num_batches
        log_entry = (f"Decom Epoch [{epoch}/{num_epochs_decom}], "
                    f"Avg Loss: {avg_loss}, Avg Target Loss: {avg_target_loss}\n")
        log_file.write(log_entry)
        
        decom_model.eval()
        val_target_loss = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for val_input_image, val_target_image in val_dataloader:
                val_input_image, val_target_image = val_input_image.to(device), val_target_image.to(device)
                val_R, val_L = decom_model(val_input_image)
                val_L_expanded = val_L.expand_as(val_R)

                val_loss_recon = criterion_mse(val_R * val_L_expanded, val_input_image)
                val_loss_R = 1.0 - ssim_module_rgb(val_R, val_target_image)
                val_loss_color = sum(1.0 - ssim_module_channel(val_R[:, i:i+1, :, :], val_target_image[:, i:i+1, :, :]) for i in range(3))
                val_loss_brightness_L = brightness_loss(val_L, target_mean=0.5, min_threshold=0.1, max_threshold=0.9)
                val_loss_contrast_L = contrast_loss(val_L, bins=100)
                val_loss_variation_R = variation_loss(val_R)
                val_loss_l2_R = l2_regularization(val_R)
                val_loss_channel_diversity = channel_diversity_loss(val_R)
                val_loss_tv_R = tv_loss(val_R)
                val_loss_brightness_R = brightness_constraint_R(val_R, target_mean=0.5)
                val_loss_histogram_L = histogram_loss(val_L)

                val_target_loss += (1.0 * val_loss_recon +
                                    0.0 * val_loss_R +
                                    0.0 * val_loss_color +
                                    0.0 * val_loss_contrast_L +
                                    0.0 * val_loss_histogram_L +
                                    0.0 * val_loss_variation_R +
                                    0.0 * val_loss_tv_R)
                if val_num_batches == 0:
                    vutils.save_image(val_R, f"val_decom_R_epoch_{epoch}.png", normalize=False)
                    vutils.save_image(val_L, f"val_decom_L_epoch_{epoch}.png", normalize=False)
                val_num_batches += 1

        avg_val_target_loss = val_target_loss / val_num_batches
        log_entry = (f"Validation Epoch [{epoch}/{num_epochs_decom}], "
                    f"Avg Val Target Loss: {avg_val_target_loss}, "
                    f"Val L mean: {val_L.mean().item()}, Val L var: {val_L.var().item()}, "
                    f"Val R mean: {val_R.mean().item()}, Val R var: {val_R.var().item()}, "
                    f"val_loss_recon: {val_loss_recon.item()}, "
                    f"val_loss_R: {val_loss_R.item()}, "
                    f"val_loss_color: {val_loss_color.item()}, "
                    f"val_loss_contrast_L: {val_loss_contrast_L.item()}, "
                    f"val_loss_histogram_L: {val_loss_histogram_L.item()}, "
                    f"val_loss_variation_R: {val_loss_variation_R.item()}, "
                    f"val_loss_tv_R: {val_loss_tv_R.item()}\n")
        log_file.write(log_entry)

        other_lr = other_optimizer.param_groups[0]['lr']
        recon_R_lr = recon_R_optimizer.param_groups[0]['lr']
        log_entry = f"Other Optimizer LR: {other_lr}, Recon R Optimizer LR: {recon_R_lr}\n"
        log_file.write(log_entry)

        if avg_val_target_loss < best_target_loss:
            best_target_loss = avg_val_target_loss
            counter = 0
            torch.save(decom_model.state_dict(), f"best_decom_epoch_{epoch}.pth")
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping for Decom")
            break

        decom_model.train()

    log_file.close()