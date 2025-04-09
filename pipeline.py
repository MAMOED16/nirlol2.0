import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.utils as vutils

DECOM_MODEL_PATH = "best_decom_epoch_66.pth"
ENHANCE_MODEL_PATH = "enhance_epoch_154.pth"

INPUT_DIR = "C:/NIR_LOL_C/dataset/train/low"
OUTPUT_DIR = "C:/NIR_LOL_C/out"
SIZE_X = 600
SIZE_Y = 400
# Параметры постобработки
GAMMA = 2.5  # гамма-коррекция (1.0 = нет коррекции)
SATURATION_FACTOR = 1.2  # насыщенность (1.0 = нет коррекции)
CONTRAST_FACTOR = 2.5 # контраст (1.0 = нет коррекции)
INPUT_DR_FACTOR = 10 # динамический диапазон входных изображений. меньше = агрессивнее коррекция. 10 для LOL dataset, 15-30 для более контрастных

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecomNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.conv0 = nn.Conv2d(3, channels, kernel_size, padding=4, padding_mode='replicate')
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
        )
        self.recon_R = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, 3, kernel_size, padding=1, padding_mode='replicate'),
            nn.Sigmoid()
        )
        self.recon_L = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channels, 1, kernel_size, padding=1, padding_mode='replicate'),
            nn.Sigmoid()
        )
        self.resize = nn.AdaptiveAvgPool2d((SIZE_Y, SIZE_X))
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv_layers(x)
        R = self.recon_R(x)
        L = self.recon_L(x)
        R = R + x[:, :3, :, :]
        R = torch.clamp(R, 0, 1)
        L = torch.clamp(L, 0, 1)
        R = self.resize(R)
        L = self.resize(L)
        return R, L

class EnhanceNet(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(EnhanceNet, self).__init__()
        self.conv0 = nn.Conv2d(1, channels, kernel_size, padding=1, padding_mode='replicate')
        self.bn0 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(channels)
        self.output = nn.Conv2d(channels, 1, kernel_size, padding=1, padding_mode='replicate')
        
        self.low_freq = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='replicate')
        
        self.denoise = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='replicate')
        )
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        self.resize = nn.AdaptiveAvgPool2d((SIZE_Y, SIZE_X))

    def forward(self, L):
        x = self.leaky_relu(self.bn0(self.conv0(L)))
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        raw_output = self.output(x)

        scale_factor = torch.max(torch.abs(raw_output)) + 1e-6 
        
        # scaled_output = (raw_output / (scale_factor * 0.45))
        scaled_output = (raw_output / INPUT_DR_FACTOR) + 0.1 * INPUT_DR_FACTOR
        L_enhanced = scaled_output
        
        L_low_freq = self.low_freq(L_enhanced)
        L_low_freq = self.resize(L_low_freq)
        
        L_enhanced = self.resize(L_enhanced)
        L_high_freq = L_enhanced - L_low_freq

        L_high_freq_denoised = self.denoise(L_high_freq)
        
        L_high_freq_denoised = self.resize(L_high_freq_denoised)

        output = L_low_freq + (5.2 / (4 + np.exp(0.1 * INPUT_DR_FACTOR))) * L_high_freq_denoised
        # beta = 1.5
        # output = torch.clamp(output + 1 * beta, 0, 1)
        output = torch.sigmoid(output)
        return output

# Функция для преобразования RGB в HSV
def rgb_to_hsv(rgb):
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    
    max_val, max_idx = torch.max(rgb, dim=1, keepdim=True)
    min_val = torch.min(rgb, dim=1, keepdim=True)[0]      
    delta = max_val - min_val                             
    
    h = torch.zeros_like(r)
    delta_squeezed = delta.squeeze(1)
    max_idx_squeezed = max_idx.squeeze(1)
    
    mask = (max_idx_squeezed == 0) & (delta_squeezed != 0)
    h[mask] = 60 * (((g - b) / delta_squeezed)[mask] % 6)
    
    mask = (max_idx_squeezed == 1) & (delta_squeezed != 0)
    h[mask] = 60 * (((b - r) / delta_squeezed)[mask] + 2)
    
    mask = (max_idx_squeezed == 2) & (delta_squeezed != 0)
    h[mask] = 60 * (((r - g) / delta_squeezed)[mask] + 4)
    
    h = h / 360.0
    
    s = torch.zeros_like(r) 
    max_val_squeezed = max_val.squeeze(1)
    mask = max_val_squeezed != 0
    s[mask] = (delta_squeezed / max_val_squeezed)[mask]
    
    v = max_val.squeeze(1)
    
    return torch.stack([h, s, v], dim=1)

# Функция для преобразования HSV в RGB
def hsv_to_rgb(hsv):
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    
    h = h * 360.0
    c = v * s
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
    m = v - c
    
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    
    mask = (h >= 0) & (h < 60)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0
    
    mask = (h >= 60) & (h < 120)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0
    
    mask = (h >= 120) & (h < 180)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
    
    mask = (h >= 180) & (h < 240)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
    
    mask = (h >= 240) & (h < 300)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
    
    mask = (h >= 300) & (h < 360)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
    
    r, g, b = r + m, g + m, b + m
    
    return torch.stack([r, g, b], dim=1)

# Функция для создания I_enhanced с использованием HSV
def create_enhanced_image_hsv(R_low, L_enhanced):
    hsv = rgb_to_hsv(R_low)
    h, s, _ = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    
    v = L_enhanced.squeeze(1)
    hsv_enhanced = torch.stack([h, s, v], dim=1)
    
    I_enhanced = hsv_to_rgb(hsv_enhanced)
    I_enhanced = torch.clamp(I_enhanced, 0, 1)
    return I_enhanced

# Функция для увеличения насыщенности (опционально)
def increase_saturation(image, factor=1.0):
    if factor == 1.0:
        return image
    hsv = rgb_to_hsv(image)
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    s = torch.clamp(s * factor, 0, 1)
    hsv = torch.stack([h, s, v], dim=1)
    return hsv_to_rgb(hsv)

def apply_gamma_correction(image, gamma=1.0):
    if gamma == 1.0:
        return image
    return torch.pow(image, 1.0 / gamma)

def adjust_contrast(image, contrast_factor=1.0):
    if contrast_factor == 1.0:
        return image
    adjusted = contrast_factor * (image - 0.5) + 0.5
    return torch.clamp(adjusted, 0, 1)
def to_rgb(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    return image
def save_image_grid(input_image, R_low, L_low, L_enhanced, I_enhanced, output_path):
    # Приводим все изображения к трёхканальному формату
    input_image_rgb = to_rgb(input_image) 
    R_low_rgb = to_rgb(R_low)             
    L_low_rgb = to_rgb(L_low)             
    L_enhanced_rgb = to_rgb(L_enhanced)   
    I_enhanced_rgb = to_rgb(I_enhanced)   

    images = [input_image_rgb, R_low_rgb, L_low_rgb, L_enhanced_rgb, I_enhanced_rgb]

    grid = vutils.make_grid(
        torch.cat(images, dim=0), 
        nrow=5,                   
        padding=10,               
        normalize=True            
    )

    vutils.save_image(grid, output_path)
blurrer = transforms.GaussianBlur(kernel_size=(7, 13), sigma=(7, 11))
decom_model = DecomNet().to(device)
decom_model.load_state_dict(torch.load(DECOM_MODEL_PATH, weights_only=True))
decom_model.eval()

enhance_net = EnhanceNet().to(device)
enhance_net.load_state_dict(torch.load(ENHANCE_MODEL_PATH, weights_only=True))
enhance_net.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    # transforms.Resize((400, 600)),
    transforms.ToTensor()
])

for img_name in os.listdir(INPUT_DIR):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, img_name)
        image = Image.open(img_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            R_low, L_low = decom_model(input_image)
        
        with torch.no_grad():
            L_enhanced = enhance_net(L_low)
            L_enhanced = blurrer(L_enhanced)
            L_enhanced = 0.7 * L_enhanced + 0.3 * L_low
            I_enhanced = R_low * L_enhanced

        I_enhanced = torch.pow(I_enhanced, 1.0 / GAMMA)
        I_enhanced = torch.clamp(I_enhanced, 0, 1)
        
        I_enhanced = increase_saturation(I_enhanced, SATURATION_FACTOR)
        I_enhanced = torch.clamp(I_enhanced, 0, 1)

        I_enhanced = adjust_contrast(I_enhanced, contrast_factor=CONTRAST_FACTOR)
        I_enhanced = torch.clamp(I_enhanced, 0, 1)

        grid_path = os.path.join(OUTPUT_DIR, f"grid_{img_name}")
        save_image_grid(input_image, R_low, L_low, L_enhanced, I_enhanced, grid_path)
        print(f"Сетка сохранена как: {grid_path}")
        # output_path = os.path.join(OUTPUT_DIR, f"enhanced_{img_name}")
        # vutils.save_image(I_enhanced, output_path, normalize=False)
        # print(f"Обработано изображение: {img_name}, сохранено как: {output_path}")

print("Обработка завершена!")