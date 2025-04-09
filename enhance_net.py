import torch.nn as nn
import torch
import torch.nn.functional as F

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         skip = x 
#         x = self.encoder(x)
#         x = self.decoder(x)
        
#         return torch.clamp(0.7 * x + 0.3 * skip, 0, 1)



# import torch.nn as nn

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x




# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         return x + self.conv2(self.relu(self.conv1(x)))

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(256),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return torch.clamp(x, 0, 1)


# class ResidualBlock(nn.Module):
#     """ Остаточный блок для сохранения деталей """
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         return x + self.conv2(self.relu(self.conv1(x)))

# class EnhanceNet(nn.Module):
#     def __init__(self):
#         super(EnhanceNet, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(64),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(128),
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             ResidualBlock(64),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         skip = x
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return torch.clamp(0.9 * x + 0.1 * skip, 0, 1)


# class EnhanceNet(nn.Module):
#     def __init__(self, channels=64, kernel_size=3):
#         super(EnhanceNet, self).__init__()

#         self.relu = nn.ReLU()
#         self.conv0 = nn.Conv2d(4, channels, kernel_size, padding=1, padding_mode='replicate')
        
#         # Downsampling с добавлением слоев
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode='replicate')
#         self.conv1_1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode='replicate')
#         self.conv2_1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.conv3 = nn.Conv2d(channels, channels, kernel_size, stride=2, padding=1, padding_mode='replicate')
#         self.conv3_1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')

#         # Upsampling с transposed convolutions
#         self.deconv1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=4, stride=2, padding=1)

#         # Денойзинг слой
#         self.denoise = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
        
#         self.fusion = nn.Conv2d(channels * 3, channels, kernel_size=1, padding=0, padding_mode='replicate')
#         self.output = nn.Conv2d(channels, 1, kernel_size=3, padding=1, padding_mode='replicate')
        
#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, I, R):
#         target_size = I.shape[2:]

#         x = torch.cat((R, I), dim=1)
#         x0 = self.conv0(x)
#         x1 = self.relu(self.conv1(x0))
#         x1 = self.relu(self.conv1_1(x1))
#         x2 = self.relu(self.conv2(x1))
#         x2 = self.relu(self.conv2_1(x2))
#         x3 = self.relu(self.conv3(x2))
#         x3 = self.relu(self.conv3_1(x3))

#         # Выравнивание размеров перед объединением
#         x3_upscaled = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
#         x3_up = self.relu(self.deconv1(torch.cat((x3_upscaled, x2), dim=1)))

#         x4_upscaled = F.interpolate(x3_up, size=x1.shape[2:], mode='bilinear', align_corners=False)
#         x4 = self.relu(self.deconv2(torch.cat((x4_upscaled, x1), dim=1)))

#         x5_upscaled = F.interpolate(x4, size=x0.shape[2:], mode='bilinear', align_corners=False)
#         x5 = self.relu(self.deconv3(torch.cat((x5_upscaled, x0), dim=1)))

#         # Денойзинг
#         x5 = self.relu(self.denoise(x5))

#         # Выравнивание размеров для fusion
#         x3_up_resized = F.interpolate(x3_up, size=x5.shape[2:], mode='bilinear', align_corners=False)
#         x4_resized = F.interpolate(x4, size=x5.shape[2:], mode='bilinear', align_corners=False)

#         fusion = self.fusion(torch.cat((x3_up_resized, x4_resized, x5), dim=1))
#         raw_output = self.output(fusion)
#         scaled_output = raw_output / 20.0
#         output = torch.sigmoid(scaled_output)

#         # Приводим output к исходным размерам входа
#         output = F.interpolate(output, size=target_size, mode='bilinear', align_corners=False)

#         print(f"EnhanceNet forward: input min {I.min().item()}, max {I.max().item()}")
#         print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
#         print(f"EnhanceNet output min: {output.min().item()}, max: {output.max().item()}")

#         return output


# class EnhanceNet(nn.Module):
#     def __init__(self, channels=64, kernel_size=3):
#         super(EnhanceNet, self).__init__()
#         self.conv0 = nn.Conv2d(1, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn0 = nn.BatchNorm2d(channels)
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(channels)
#         self.output = nn.Conv2d(channels, 1, kernel_size, padding=1, padding_mode='replicate')
#         self.relu = nn.ReLU()
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#         self.resize = nn.AdaptiveAvgPool2d((340, 512))

#     def forward(self, L):
#         x = self.relu(self.bn0(self.conv0(L)))
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         raw_output = self.output(x)
#         # Масштабирование перед Sigmoid
#         scale_factor = torch.max(torch.abs(raw_output)) + 1e-6  # Добавляем малую константу для стабильности
#         scaled_output = raw_output / (scale_factor * 0.2)
#         output = torch.sigmoid(scaled_output)
        
#         # print(f"EnhanceNet forward: input min {L.min().item()}, max {L.max().item()}")
#         # print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
#         # print(f"EnhanceNet output min: {output.min().item()}, max: {output.max().item()}")
#         output = self.resize(output)
#         return output
    

# class EnhanceNet(nn.Module):
#     def __init__(self, channels=64, kernel_size=5, dilation=2):
#         super(EnhanceNet, self).__init__()
#         # Входной слой с увеличенным ядром и dilation
#         self.conv0 = nn.Conv2d(1, channels, kernel_size, padding=(kernel_size//2)*dilation, dilation=dilation, padding_mode='replicate')
#         self.bn0 = nn.BatchNorm2d(channels)
        
#         # Второй слой с dilation
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2)*dilation, dilation=dilation, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(channels)
        
#         # Третий слой с dilation
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size//2)*dilation, dilation=dilation, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(channels)
        
#         # Сглаживающий слой с большим ядром
#         self.smooth = nn.Conv2d(channels, channels, kernel_size=7, padding=3, padding_mode='replicate')
#         self.bn_smooth = nn.BatchNorm2d(channels)
        
#         # Выходной слой
#         self.output = nn.Conv2d(channels, 1, kernel_size=3, padding=1, padding_mode='replicate')
        
#         # Мягкая активация
#         self.leaky_relu = nn.LeakyReLU(0.1)
#         self.resize = nn.AdaptiveAvgPool2d((340, 512))
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, L):
#         x = self.leaky_relu(self.bn0(self.conv0(L)))
#         x = self.leaky_relu(self.bn1(self.conv1(x)))
#         x = self.leaky_relu(self.bn2(self.conv2(x)))
#         x = self.leaky_relu(self.bn_smooth(self.smooth(x)))
#         raw_output = self.output(x)
#         scaled_output = raw_output / 20.0
#         output = torch.sigmoid(scaled_output)
        
#         print(f"EnhanceNet forward: input min {L.min().item()}, max {L.max().item()}")
#         print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
#         print(f"EnhanceNet output min: {output.min().item()}, max: {output.max().item()}")
#         self.resize(output)
#         return output



# class EnhanceNet(nn.Module):
#     def __init__(self, channels=64, kernel_size=3):
#         super(EnhanceNet, self).__init__()
#         self.conv0 = nn.Conv2d(1, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn0 = nn.BatchNorm2d(channels)
        
#         # Пулинг для захвата глобального контекста
#         self.pool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(channels)
        
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=1, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(channels)
        
#         # Сглаживающий слой
#         self.smooth = nn.Conv2d(channels, channels, kernel_size=7, padding=3, padding_mode='replicate')
#         self.bn_smooth = nn.BatchNorm2d(channels)
    
#         self.output = nn.Conv2d(channels, 1, kernel_size=3, padding=1, padding_mode='replicate')
        
#         self.leaky_relu = nn.LeakyReLU(0.1)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, L):
#         x = self.leaky_relu(self.bn0(self.conv0(L)))
        
#         # Пулинг и восстановление размера
#         x_pooled = self.pool(x)
#         x_pooled = self.leaky_relu(self.bn1(self.conv1(x_pooled)))
#         x_pooled = F.interpolate(x_pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
        
#         # Объединение с исходными признаками
#         x = x + x_pooled
        
#         x = self.leaky_relu(self.bn2(self.conv2(x)))
#         x = self.leaky_relu(self.bn_smooth(self.smooth(x)))
#         raw_output = self.output(x)
#         scaled_output = raw_output / 20.0
#         output = torch.sigmoid(scaled_output)
        
#         print(f"EnhanceNet forward: input min {L.min().item()}, max {L.max().item()}")
#         print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
#         print(f"EnhanceNet output min: {output.min().item()}, max: {output.max().item()}")
        
#         return output


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
        
        # Слой для выделения НЧ
        self.low_freq = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='replicate')
        
        # Денойзинг для ВЧ
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

        self.resize = nn.AdaptiveAvgPool2d((340, 512))

    def forward(self, L):
        skip = L
        x = self.leaky_relu(self.bn0(self.conv0(L)))
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        raw_output = self.output(x)
        # print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
        # Масштабирование перед Sigmoid
        # scale_factor = torch.max(torch.abs(raw_output)) + 1e-6 
        scaled_output = (raw_output / 5) - 2.2
        L_enhanced = scaled_output
        
        # Частотное разложение
        L_low_freq = self.low_freq(L_enhanced)
        L_low_freq = self.resize(L_low_freq)
        
        L_enhanced = self.resize(L_enhanced)
        L_high_freq = L_enhanced - L_low_freq
        # print(f"EnhanceNet L_enhanced (before denoise) min: {L_enhanced.min().item()}, max: {L_enhanced.max().item()}")
        # Денойзинг ВЧ
        L_high_freq_denoised = self.denoise(L_high_freq)
        L_high_freq_denoised = self.resize(L_high_freq_denoised)

        output = L_low_freq + 0.5 * L_high_freq_denoised
        # print(f"EnhanceNet output (after denoise) min: {output.min().item()}, max: {output.max().item()}")
        # beta = 1.5
        # output = torch.clamp(output + 1 * beta, 0, 1)
        output = torch.sigmoid(output)
        # print(f"EnhanceNet output (after sigmoid) min: {output.min().item()}, max: {output.max().item()}")
    

    


# class EnhanceNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, base_channels=32):
#         super(EnhanceNet, self).__init__()
        
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels),
#             nn.LeakyReLU(0.1)
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 340x512 -> 170x256
        
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.LeakyReLU(0.1)
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 170x256 -> 85x128
        
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.LeakyReLU(0.1)
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 85x128 -> 43x64
        
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 8),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 8),
#             nn.LeakyReLU(0.1)
#         )
        
#         # Декодер
#         self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)  # 43x64 -> 86x128
#         self.dec3 = nn.Sequential(
#             nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 4),
#             nn.LeakyReLU(0.1)
#         )
        
#         self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)  # 86x128 -> 172x256
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels * 2),
#             nn.LeakyReLU(0.1)
#         )
        
#         self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # 172x256 -> 344x512
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, padding_mode='replicate'),
#             nn.BatchNorm2d(base_channels),
#             nn.LeakyReLU(0.1)
#         )
        
#         self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate')
        
#         # Сглаживающий слой для уменьшения артефактов
#         self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, padding_mode='replicate')
        
#         self.resize = nn.AdaptiveAvgPool2d((340, 512))
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, L):

#         e1 = self.enc1(L)  # [batch_size, 32, 340, 512]
#         e2 = self.enc2(self.pool1(e1))  # [batch_size, 64, 170, 256]
#         e3 = self.enc3(self.pool2(e2))  # [batch_size, 128, 85, 128]
#         bottleneck = self.bottleneck(self.pool3(e3))  # [batch_size, 256, 43, 64]
        
#         d3 = self.upconv3(bottleneck)  # [batch_size, 128, 86, 128]
#         e3 = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
#         d3 = self.dec3(torch.cat([d3, e3], dim=1))  # [batch_size, 128, 86, 128]
        
#         d2 = self.upconv2(d3)  # [batch_size, 64, 172, 256]

#         e2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
#         d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [batch_size, 64, 172, 256]
        
#         d1 = self.upconv1(d2)  # [batch_size, 32, 344, 512]

#         e1 = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
#         d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [batch_size, 32, 344, 512]
        
#         raw_output = self.final_conv(d1)  # [batch_size, 1, 344, 512]
#         print(f"EnhanceNet raw output min: {raw_output.min().item()}, max: {raw_output.max().item()}")
#         # Масштабирование перед Sigmoid
#         scale_factor = torch.max(torch.abs(raw_output)) + 1e-6
#         # scaled_output = raw_output / (scale_factor * 0.7)
#         scaled_output = (raw_output / 10) + 1.3
#         L_enhanced = scaled_output
        
#         # Сглаживание
#         print(f"EnhanceNet L_enhanced (before smooth) min: {L_enhanced.min().item()}, max: {L_enhanced.max().item()}")
#         output = self.smooth(L_enhanced)
#         print(f"EnhanceNet output (after smooth) min: {output.min().item()}, max: {output.max().item()}")

#         beta = 1.5
#         L_enhanced = torch.clamp(L_enhanced * beta, 0, 1)
#         output = self.resize(L_enhanced)  # Приведение к [batch_size, 1, 340, 512]

#         print(f"EnhanceNet forward: input min {L.min().item()}, max {L.max().item()}")

#         return output