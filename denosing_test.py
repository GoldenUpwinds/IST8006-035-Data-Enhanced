import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio

from models import SimpleAutoencoder as Net

load_path = os.path.join("models", "Encoder_3000.pth")

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = os.listdir(os.path.join(root_dir, 'NOISE'))

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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

# 设置测试集路径
test_dataset_path = 'D:\Project\Denoising\dataset\\test'

# 创建自定义测试数据集和数据加载器
test_dataset = TestDataset(root_dir=test_dataset_path)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型
model = Net()
model = model.to(device)

# 加载之前训练好的模型权重
model.load_state_dict(torch.load(load_path))
model.eval()  # 将模型设置为评估模式

# 在测试集上计算PSNR
total_psnr = 0
with torch.no_grad():
    for batch in test_data_loader:
        input_images, gt_images = batch
        input_images = input_images.to(device)
        gt_images = gt_images.to(device)
        outputs = model(input_images)
        for i in range(outputs.size(0)):
            psnr = peak_signal_noise_ratio(gt_images[i].cpu().numpy(), outputs[i].cpu().numpy())
            total_psnr += psnr

average_psnr = total_psnr / len(test_data_loader.dataset)
print(f'Average PSNR on Test Set: {average_psnr:.4f}')
