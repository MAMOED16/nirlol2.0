import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils as vutils
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import SSIM
from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet
import numpy as np

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
    
DECOM_MODEL_PATH = "best_decom_epoch_66.pth"
OUTPUT_DIR = "enhance_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
LOG_FILE_PATH = "enhance_training_log.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((340, 512)),
    transforms.ToTensor(),
])
def tv_loss(tensor):
    batch_size = tensor.size(0)
    h = tensor.size(2)
    w = tensor.size(3)
    tv_h = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).sum()
    tv_w = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).sum()
    return (tv_h + tv_w) / (batch_size * h * w)
def log_histogram(tensor, name, epoch, batch, log_file):
    tensor = torch.clamp(tensor, 0, 1)
    tensor_np = tensor.detach().cpu().flatten().numpy()
    hist, _ = np.histogram(tensor_np, bins=100, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-6)
    log_entry = f"{name} Histogram Epoch [{epoch}/{num_epochs_enhance}], Batch [{batch+1}/{len(dataloader)}]: {hist.tolist()}\n"
    log_file.write(log_entry)

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
    decom_model.load_state_dict(torch.load(DECOM_MODEL_PATH))
    decom_model.eval()

    enhance_net = EnhanceNet().to(device)
    optimizer = optim.Adam(enhance_net.parameters(), lr=1e-3)
    patience = 100

    criterion_mse = nn.MSELoss()

    num_epochs_enhance = 500
    best_target_loss = float('inf')
    counter = 0
    log_file = open(LOG_FILE_PATH, "w")

    for epoch in range(num_epochs_enhance):
        enhance_net.train()
        total_loss = 0.0
        total_target_loss = 0.0
        num_batches = 0
        
        for i, (input_image, target_image) in enumerate(dataloader):
            input_image, target_image = input_image.to(device), target_image.to(device)

            with torch.no_grad():
                R_low, L_low = decom_model(input_image)
            
            optimizer.zero_grad()
            
            L_enhanced = enhance_net(L_low)
            I_enhanced = R_low * L_enhanced
            gamma = 1.5
            I_enhanced = torch.pow(I_enhanced, 1.0 / gamma)
            I_enhanced = torch.clamp(I_enhanced, 0, 1)
            
            log_histogram(L_enhanced, "L_enhanced", epoch, i, log_file)
            log_histogram(I_enhanced, "I_enhanced", epoch, i, log_file)
            
            loss_recon = criterion_mse(I_enhanced, target_image)
            loss_illumination_smooth = tv_loss(L_enhanced)
            
            l2_reg = 0.0
            for param in enhance_net.parameters():
                l2_reg += torch.norm(param, p=2)
            
            loss = 10 * loss_recon + 0.001 * loss_illumination_smooth + 0.001 * l2_reg
            target_loss = loss_recon  # Для валидации используем только loss_recon
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(enhance_net.parameters(), max_norm=5.0)
            log_entry = f"Gradient norm: {grad_norm.item()}\n"
            log_file.write(log_entry)
            
            # Логирование градиентов
            for name, param in enhance_net.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    log_entry = f"Gradient mean for {name}: {grad_mean}\n"
                    log_file.write(log_entry)
            
            
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_target_loss += target_loss.item()
            num_batches += 1
            
            I_enhanced_mins = torch.amin(I_enhanced, dim=[0, 2, 3]).tolist()
            I_enhanced_maxs = torch.amax(I_enhanced, dim=[0, 2, 3]).tolist()
            # log_entry = (f"Epoch [{epoch}/{num_epochs_enhance}], Batch [{i+1}/{len(dataloader)}], "
            #             f"I_enhanced [R,G,B] min: {I_enhanced_mins}, max: {I_enhanced_maxs}, "
            #             f"L_enhanced min: {L_enhanced.min().item()}, max: {L_enhanced.max().item()}, "
            #             f"loss_recon: {loss_recon.item()}, "
            #             f"loss_illumination_smooth: {loss_illumination_smooth.item()}, "
            #             f"l2_reg: {l2_reg.item()}\n")
            # log_file.write(log_entry)
            
        
        avg_loss = total_loss / num_batches
        avg_target_loss = total_target_loss / num_batches
        print(f"Enhance Epoch [{epoch}/{num_epochs_enhance}], Avg Loss: {avg_loss}, Avg Target Loss: {avg_target_loss}")
        vutils.save_image(I_enhanced, os.path.join(OUTPUT_DIR, f"enhanced_epoch_{epoch}.png"), normalize=False)
        vutils.save_image(L_enhanced, os.path.join(OUTPUT_DIR, f"L_enhanced_epoch_{epoch}.png"), normalize=False)
        # log_entry = (f"Enhance Epoch [{epoch}/{num_epochs_enhance}], "
        #             f"Avg Loss: {avg_loss}, Avg Target Loss: {avg_target_loss}\n")
        # log_file.write(log_entry)
        torch.save(enhance_net.state_dict(), os.path.join(OUTPUT_DIR, f"enhance_epoch_{epoch}.pth"))

        # Валидационный цикл
        enhance_net.eval()
        val_target_loss = 0.0
        val_num_batches = 0
        with torch.no_grad():
            for val_input_image, val_target_image in val_dataloader:
                val_input_image, val_target_image = val_input_image.to(device), val_target_image.to(device)
                
                val_R_low, val_L_low = decom_model(val_input_image)
            
                val_L_enhanced = enhance_net(val_L_low)
                val_I_enhanced = val_R_low * val_L_enhanced
                val_I_enhanced = torch.pow(val_I_enhanced, 1.0 / gamma)
                val_I_enhanced = torch.clamp(val_I_enhanced, 0, 1)
                if val_num_batches == 0:
                    vutils.save_image(val_I_enhanced, os.path.join(OUTPUT_DIR, f"val_enhanced_epoch_{epoch}.png"), normalize=False)
                    vutils.save_image(val_L_enhanced, os.path.join(OUTPUT_DIR, f"val_L_enhanced_epoch_{epoch}.png"), normalize=False)

                val_loss_recon = criterion_mse(val_I_enhanced, val_target_image)
                val_loss_illumination_smooth = tv_loss(val_L_enhanced)
                
                val_target_loss += 10 * val_loss_recon.item()
                val_num_batches += 1
                

                # log_entry = (f"Validation Epoch [{epoch}/{num_epochs_enhance}], "
                #             f"Val L_enhanced mean: {val_L_enhanced.mean().item()}, "
                #             f"Val L_enhanced var: {val_L_enhanced.var().item()}, "
                #             f"Val I_enhanced mean: {val_I_enhanced.mean().item()}, "
                #             f"Val I_enhanced var: {val_I_enhanced.var().item()}, "
                #             f"val_loss_recon: {val_loss_recon.item()}, "
                #             f"val_loss_illumination_smooth: {val_loss_illumination_smooth.item()}\n")
                # log_file.write(log_entry)
        
        avg_val_target_loss = val_target_loss / val_num_batches
        print(f"Validation Epoch [{epoch}/{num_epochs_enhance}], Avg Val Target Loss: {avg_val_target_loss}")
        
        log_entry = f"Validation Epoch [{epoch}/{num_epochs_enhance}], Avg Val Target Loss: {avg_val_target_loss}\n"
        log_file.write(log_entry)
        
        lr = optimizer.param_groups[0]['lr']
        log_entry = f"Optimizer LR: {lr}\n"
        log_file.write(log_entry)
        
        # Сохранение лучшей модели
        if avg_val_target_loss < best_target_loss:
            best_target_loss = avg_val_target_loss
            counter = 0
            torch.save(enhance_net.state_dict(), os.path.join(OUTPUT_DIR, f"best_enhance_epoch_{epoch}.pth"))
        else:
            counter += 1
        
        if counter >= patience:
            print("Early stopping for Enhance")
            break
        
        enhance_net.train()

    log_file.close()