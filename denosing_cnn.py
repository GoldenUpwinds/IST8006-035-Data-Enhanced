import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio

from models import SimpleAutoencoder as Net

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = os.listdir(os.path.join(root_dir, 'NOISE'))

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        noise_image_path = os.path.join(self.root_dir, 'NOISE', image_name)
        gt_image_path = os.path.join(self.root_dir, 'GT', image_name)

        noise_image = Image.open(noise_image_path).convert('RGB')
        gt_image = Image.open(gt_image_path).convert('RGB')

        noise_image = self.transform(noise_image)
        gt_image = self.transform(gt_image)

        return noise_image, gt_image

# 设置数据集路径
dataset_path = 'D:\Project\Denoising\dataset\\train'

# 创建自定义数据集和数据加载器
custom_dataset = CustomDataset(root_dir=dataset_path)
data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# 初始化模型、损失函数和优化器
# model = SimpleCNN(input_channels=3)
model = Net()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

max_psnr = 0.0
best_model = None

# 训练模型
num_epochs = 30
for epoch in range(num_epochs):
    for batch in data_loader:
        input_images, gt_images = batch
        input_images = input_images.to(device)
        gt_images = gt_images.to(device)

        # 前向传播
        outputs = model(input_images)

        # 计算损失
        loss = criterion(outputs, gt_images)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在每个 epoch 结束时计算并打印平均 PSNR
    total_psnr = 0
    with torch.no_grad():
        for batch in data_loader:
            input_images, gt_images = batch
            input_images = input_images.to(device)
            gt_images = gt_images.to(device)
            outputs = model(input_images)
            for i in range(outputs.size(0)):
                psnr = peak_signal_noise_ratio(gt_images[i].cpu().numpy(), outputs[i].cpu().numpy())
                total_psnr += psnr
    average_psnr = total_psnr / len(data_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Average PSNR: {average_psnr:.4f}')

    # 保存 PSNR 最大的模型
    if average_psnr > max_psnr:
        max_psnr = average_psnr
        best_model = model.state_dict()

# 保存模型
save_path = os.path.join("models", "simple_cnn_model.pth")
torch.save(best_model, save_path)
