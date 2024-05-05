import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
import torchvision.models as models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from street_tryon_benchmark.dataset import GeneralTryOnDataset



def get_dataset_by_task(task):
    if task == 'shop2model':
        config_path = "street_tryon_benchmark/configs/shop2model.yaml"
    elif task == 'shop2street':
        config_path = "street_tryon_benchmark/configs/shop2street.yaml"
    elif task == 'model2model':
        config_path = "street_tryon_benchmark/configs/model2model.yaml"
    elif task == 'model2street':
        config_path = "street_tryon_benchmark/configs/model2street.yaml"
    elif task == 'street2street-top':
        config_path = "street_tryon_benchmark/configs/street2street_top.yaml"
    elif task == 'street2street-dress':
        config_path = "street_tryon_benchmark/configs/street2street_dress.yaml"
    else:
        raise NotImplementedError


    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    return GeneralTryOnDataset(".", config=data_config, split='test')

# create dataset for street2street task
dataset = get_dataset_by_task('street2street-top')

# check data
curr = dataset[0]

# get person-related data
pimg, piuv, pseg = curr['pimg'], curr['piuv'], curr['pseg']

# get garment-related data
gimg, giuv, gseg = curr['gimg'], curr['giuv'], curr['gseg']


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DensePoseEncoder(nn.Module):
    def __init__(self):
        super(DensePoseEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SqueezeExcitation(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeExcitation(256)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, groups=256),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SqueezeExcitation(128)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        return x

    

class StyleCodeGenerator(nn.Module):
    def __init__(self):
        super(StyleCodeGenerator, self).__init__()
        self.shared_encoder = nn.Sequential(
            #inchannel,outchannel
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )


    def forward(self, top, hair, pants, skirt):
        z_top = self.shared_encoder(top)
        z_hair = self.shared_encoder(hair)
        z_pants = self.shared_encoder(pants)
        z_skirt = self.shared_encoder(skirt)
        z = torch.cat((z_top, z_hair, z_pants, z_skirt), dim=1)
        #z = self.additional_layers(z)

        z = F.dropout(z, 0.5)
        return z
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    

class StyleGAN(nn.Module):
    def __init__(self):
        super(StyleGAN, self).__init__()
        self.initial_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(2688, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024)
        )

        self.attention1 = SelfAttention(1024)
        
        self.middle_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.attention2 = SelfAttention(512)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((512, 320))

    def forward(self, style_code, dense_pose_features):

        style_code_expanded = style_code.unsqueeze(-1).unsqueeze(-1)
        style_code_expanded = style_code_expanded.expand(-1, -1, dense_pose_features.size(2), dense_pose_features.size(3))


        combined_features = torch.cat((dense_pose_features, style_code_expanded), dim=1)
        #initial_feature = self.initial_layer(combined_features)
        combined_features_upsampled = F.interpolate(combined_features, scale_factor=2, mode='nearest')

        initial_feature = self.initial_layer(combined_features_upsampled)        

        initial_feature = self.attention1(initial_feature)


        res_feature = self.res_blocks(initial_feature)
        

        middle_feature = self.middle_layer(res_feature)
        


        adapted_feature = self.adaptive_pool(middle_feature)

        # 최종 RGB 이미지 generate
        rgb_image = self.to_rgb(adapted_feature)
        rgb_image = (rgb_image + 1) / 2  #range  to [0,1]
        return rgb_image

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out


class TryOnParseEstimator(nn.Module):
    def __init__(self, mode='train'):
        super(TryOnParseEstimator, self).__init__()
        self.mode = mode
        self.dense_pose_encoder = DensePoseEncoder()
        self.style_code_generator = StyleCodeGenerator()
        self.stylegan = StyleGAN()

    def forward(self, pimg, piuv, pseg, gimg, giuv, gseg):
        # Parse segmentation
        if pseg.dim() == 3:
            pseg = pseg.unsqueeze(1)
        if pseg.size(1) == 1:
            pseg = pseg.repeat(1, 3, 1, 1)
        
        # Extract segments
        hair_mask = (pseg == 1).float()
        hair_segment = hair_mask * pimg

        top_mask = (pseg == 2).float()
        top_segment = top_mask * pimg

        pants_mask = (pseg == 3).float()
        pants_segment = pants_mask * pimg

        skirt_mask = (pseg == 4).float()
        skirt_segment = skirt_mask * pimg

        dense_pose_feature = self.dense_pose_encoder(piuv)

        if self.mode == 'train':
            style_codes = self.style_code_generator(top_segment, hair_segment, pants_segment, skirt_segment)
        elif self.mode == 'test':
            # Test mode: swap top segment with garment image's top
            g_top_segment = (gseg == 2).float() * gimg 
            style_codes = self.style_code_generator(g_top_segment, hair_segment, pants_segment, skirt_segment)
        
        if gimg.dim() == 3:
            gimg = gimg.unsqueeze(0)
        
        gimg_style_code = self.style_code_generator.shared_encoder(gimg)
        
        if gimg_style_code.dim() == 1:
            gimg_style_code = gimg_style_code.unsqueeze(0)
        
        combined_style_code = torch.cat((style_codes, gimg_style_code), dim=1)
        
        tryon_image = self.stylegan(combined_style_code, dense_pose_feature)

        return tryon_image


tryon_estimator = TryOnParseEstimator()

#train_loader = DataLoader(dataset, batch_size=6, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델 설정
model = TryOnParseEstimator().to(device)




def save_model(model, epoch, save_dir='saved_models'):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), save_path)



import matplotlib.pyplot as plt
import os
def visualize_output(input_image, generated_image, target_image, epoch, mode, directory='output_visualizations'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(input_image.detach().cpu().permute(1, 2, 0))
    ax[0].set_title('Input Image')
    ax[1].imshow(generated_image.detach().cpu().permute(1, 2, 0))
    ax[1].set_title('Generated Image')
    ax[2].imshow(target_image.detach().cpu().permute(1, 2, 0))
    ax[2].set_title('Target Image (Original)')
    ax[0].axis('off'); ax[1].axis('off'); ax[2].axis('off')
    plt_path = os.path.join(directory, f'{mode}_epoch_{epoch}.png')
    plt.savefig(plt_path)
    plt.close()

def overfit_and_test(model, train_inputs, epochs, optimizer, criterion, scheduler):
    pimg, piuv, pseg, gimg, giuv, gseg = train_inputs

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(pimg, piuv, pseg, gimg, giuv, gseg)
        loss = criterion(outputs, pimg)  # 원본 이미지를 타겟으로 사용
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')
            visualize_output(pimg[0], outputs[0], pimg[0], epoch, 'train')

        if epoch % 60 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(pimg, piuv, pseg, gimg, giuv, gseg)
                visualize_output(pimg[0], test_outputs[0], pimg[0], epoch, 'test')

    print("Training completed.")

# 데이터 로드 및 실행 코드 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TryOnParseEstimator(mode='train').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 데이터 로드
dataset = get_dataset_by_task('street2street-top')
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
train_batch = next(iter(data_loader))
train_inputs = [train_batch[k].to(device) for k in ['pimg', 'piuv', 'pseg', 'gimg', 'giuv', 'gseg']]

# 모델 학습 및 테스트 실행
overfit_and_test(model, train_inputs, 300, optimizer, criterion, scheduler)