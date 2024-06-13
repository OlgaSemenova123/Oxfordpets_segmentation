# Oxfordpets_segmentation
Задание из курса https://lab.karpov.courses/. Датасет - OxfordIIITPet
Модель создана как аналог UNET по 5 слоев слева и 5 слоев справа, внизу bottleneck. Внизу представлена логика работы модели:
down1										up1
	down2								up2	
		down3						up3		
			down4				up4			
				down5		up5				
					bottleneck					
![image](https://github.com/OlgaSemenova123/Oxfordpets_segmentation/assets/157280225/a806014a-2b0e-44a9-bd00-dbc88feedce1)

Сверточный слой conv_plus_conv принимает на вход и выход количество каналов (base_channels). Это гиперпараметр, в моделе - 32 показал хорошую скорость обучения. Для сверточного слоя реализованы 2 нормализации.
С понижением слоя уменьшается картинка, потом она восстанавливается до исходной, путем конкатенации соответствующих тензоров слева и справа.
Анализ изменения размерности рисунка и каналов:

residual1 = self.down1(x)  # x.shape: (N, N, 3) -> (N, N, base_channels)
x = self.downsample(residual1)  # x.shape: (N, N, base_channels) -> (N // 2, N // 2, base_channels)

residual2 = self.down2(x)  # x.shape: (N // 2, N // 2, base_channels) -> (N // 2, N // 2, base_channels * 2)
x = self.downsample(residual2)  # x.shape: (N // 2, N // 2, base_channels * 2) -> (N // 4, N // 4, base_channels * 2)

residual3 = self.down3(x) ### (N // 4, N // 4, base_channels * 2) -> (N // 4, N // 4, base_channels * 4)
x = self.downsample(residual3) ### (N // 4, N // 4, base_channels * 4) -> (N // 8, N // 8, base_channels * 4)
        
residual4 = self.down4(x) ### (N // 8, N // 8, base_channels * 4) -> (N // 8, N // 8, base_channels * 8)
x = self.downsample(residual4) ### (N // 8, N // 8, base_channels * 8) -> (N // 16, N // 16, base_channels * 8)
        
residual5 = self.down5(x) ### (N // 16, N // 16, base_channels * 8) -> (N // 16, N // 16, base_channels * 16)
x = self.downsample(residual5) ### (N // 16, N // 16, base_channels * 16) -> (N // 32, N // 32, base_channels * 16)
        
x = self.bottleneck(x) # (N // 32, N // 32, base_channels * 16) -> (N // 32, N // 32, base_channels * 16)
        
x = nn.functional.interpolate(x, scale_factor=2) ### (N // 32, N // 32, base_channels * 16) -> (N // 16, N // 16, base_channels * 16)
x = torch.cat((x, residual5), dim=1) ### (N // 16, N // 16, base_channels * 16) -> (N // 16, N // 16, base_channels * 32)
x = self.up5(x) ### (N // 16, N // 16, base_channels * 32) -> (N // 16, N // 16, base_channels * 8)
        
x = nn.functional.interpolate(x, scale_factor=2) ### (N // 16, N // 16, base_channels * 8) -> (N // 8, N // 8, base_channels * 8)
x = torch.cat((x, residual4), dim=1) ### (N // 8, N // 8, base_channels * 8) -> (N // 8, N // 8, base_channels * 16)
x = self.up4(x) ### (N // 8, N // 8, base_channels * 16) -> (N // 8, N // 8, base_channels * 4)

x = nn.functional.interpolate(x, scale_factor=2) ### (N // 8, N // 8, base_channels * 4) -> (N // 4, N // 4, base_channels * 4)
x = torch.cat((x, residual3), dim=1) ### (N // 4, N // 4, base_channels * 4) -> (N // 4, N // 4, base_channels * 8)
x = self.up3(x) ### (N // 4, N // 4, base_channels * 8) -> (N // 4, N // 4, base_channels * 2)

x = nn.functional.interpolate(x, scale_factor=2) # (N // 4, N // 4, base_channels * 2) -> (N // 2, N // 2, base_channels * 2)
x = torch.cat((x, residual2), dim=1) # (N // 2, N // 2, base_channels * 2) -> (N // 2, N // 2, base_channels * 4)
x = self.up2(x) # (N // 2, N // 2, base_channels * 4) -> (N // 2, N // 2, base_channels)

x = nn.functional.interpolate(x, scale_factor=2) # (N // 2, N // 2, base_channels * 2) -> (N , N , base_channels)
x = torch.cat((x, residual1), dim=1) # (N , N , base_channels) -> (N , N , base_channels * 2)
x = self.up1(x) # (N , N , base_channels * 2) -> (N , N , base_channels) 

Все итоговые графики и результаты можно посмотреть в сохраненном ноутбуке. Итоговая Accuracy составила 88.1%
