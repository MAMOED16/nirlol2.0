import torch
import torch.nn as nn
import torch.nn.functional as F

# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#         )
#         self.reflectance = nn.Conv2d(256, 3, kernel_size=3, padding=1) 

#         self.illumination = nn.Conv2d(256, 1, kernel_size=3, padding=1) 


#     def forward(self, x):
#         features = self.encoder(x)
#         reflectance = torch.sigmoid(self.reflectance(features))
#         illumination = torch.sigmoid(self.illumination(features))
#         illumination = torch.clamp(illumination, 0.1, 0.9)  # Не даём стать чисто белым или чёрным
#         return reflectance, illumination


# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#         )
#         self.reflectance = nn.Sequential(
#             nn.Conv2d(256, 3, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#         self.illumination = nn.Sequential(
#             nn.Conv2d(256, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         features = self.encoder(x)
#         reflectance = self.reflectance(features)
#         illumination = self.illumination(features)

#         reflectance = torch.clamp(reflectance, 0.05, 1.0)

#         return reflectance, illumination



# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         return x + self.conv2(self.relu(self.conv1(x)))

# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(64),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128),
#         )
#         self.decoder_reflectance = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(64),
#             nn.Conv2d(64, 3, kernel_size=3, padding=1),
#             nn.Sigmoid(),  # Reflectance теперь в [0,1]
#         )
#         self.decoder_illumination = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid(),  # Illumination в [0,1]
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         R = self.decoder_reflectance(x)
#         I = self.decoder_illumination(x)
#         return R, I


# class DecomNet(nn.Module):
#     def __init__(self):
#         super(DecomNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#         )
#         self.reflectance = nn.Sequential(
#             nn.Conv2d(256, 3, kernel_size=3, padding=1),
#             nn.BatchNorm2d(3),
#             # nn.Sigmoid() 
#             nn.ReLU()
#         )
#         self.illumination = nn.Sequential(
#             nn.Conv2d(256, 1, kernel_size=3, padding=1),
#             # nn.Softplus()
#             nn.ReLU()
#         )

#     def forward(self, x):
#         features = self.encoder(x)
#         reflectance = self.reflectance(features)
#         illumination = self.illumination(features)


#         reflectance = reflectance / (reflectance.max() + 1e-6)  # Нормализация Reflectance
#         illumination = illumination / (illumination.max() + 1e-6)  # Нормализация Illumination

#         return reflectance, illumination



# class DecomNet(nn.Module):
#     def __init__(self, channels=64, kernel_size=3):
#         super(DecomNet, self).__init__()


#         self.conv0 = nn.Conv2d(4, channels, kernel_size * 3, padding=4, padding_mode='replicate')


#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             # Добавляем residual блок
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#         )


#         # self.recon = nn.Sequential(
#         #     nn.Conv2d(channels, 4, kernel_size, padding=1, padding_mode='replicate'),
#         #     nn.BatchNorm2d(4)
#         # )

#         self.recon_R = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, 3, kernel_size, padding=1, padding_mode='replicate'),
#             nn.Sigmoid()
#         )

#         self.recon_L = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate'),
#             nn.ReLU(),
#             nn.Conv2d(channels, 1, kernel_size, padding=1, padding_mode='replicate'),
#             nn.Sigmoid()
#         )
#     def forward(self, input_img):
#             input_max = torch.max(input_img, dim=1, keepdim=True)[0]
#             input_cat = torch.cat((input_max, input_img), dim=1)
#             features = self.conv0(input_cat)
#             residual = features
#             features = self.conv_layers(features)
#             features = features + residual
#             R = self.recon_R(features)
#             L = self.recon_L(features)
#             return R, L

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
        self.resize = nn.AdaptiveAvgPool2d((340, 512))
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