import time
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from dataset import ColorizationDataset
from model import PatchDiscriminator, ResUnet
def load_data(dataset, batch_size=1, shuffle=True, subset_indices=None, num_workers=0, pin_memory=False):
    """载入指定的 :class:`torch.utils.data.dataset` ，允许设定子集编号，载入部分"""
    if subset_indices is not None:
        subset = Subset(dataset, subset_indices)
    else:
        subset = dataset
    return DataLoader(subset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)

epochs = 200
batch_size = 1
G_in_channels = 1   # 输入L通道
G_out_channels = 2  # 输入AB通道
D_in_channels = 3   # 输入L+AB通道的拼接


# 优化器参数
# https://arxiv.org/pdf/1611.07004.pdf
# 3.3节，第一段
lr_G = 2e-4
lr_D = 2e-4
beta1 = 0.5
beta2 = 0.999
L1_Lambda = 100

# 单通道归一化参数
normalize_mean = (0.5,)
normalize_std = (0.5,)
image_size = 256

def test(img_count, i, seed=123):
    import numpy as np
    import torch
    from torchvision import transforms
#     from scipy.misc import *
    import cv2
    import matplotlib.pyplot as plt

    model_in_channels = 1  # 输入L通道
    model_out_channels = 2  # 输入AB通道
    image_size = 256 # 模型输出的图片大小
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 这里换成你的模型
    model = ResUnet(G_in_channels, G_out_channels).to(device)
    model.eval()
    model.load_state_dict(torch.load(r'/root/autodl-tmp/GAN/Model_noval/netG_epoch_'+str(i)+'.pth', map_location=device))
    
    # 选择测试的图片集合
    # (路径， 变换， 数量， 随机种子=123)
    test_set = ColorizationDataset('/root/autodl-tmp/coco3200/test2017', 
                                    transforms.Compose([transforms.Normalize(mean=normalize_mean * 3, std=normalize_std * 3), 
                                                        transforms.Resize((image_size, image_size))]),
                                    img_count,
                                    seed)
    test_data = load_data(test_set, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)

    # 针对 mean=0.5, std=0.5 的反归一化
    l_un_normalize_transform = transforms.Normalize(mean=-1.0, std=2.0)
    ab_un_normalize_transform = transforms.Normalize(mean=(-1.0, -1.0), std=(2.0, 2.0))

    
    def concat_l_ab(l, ab):
        img_lab = torch.cat([l, ab], dim=0)
        img_lab = img_lab.cpu().permute(1, 2, 0).numpy() * 255
        img_lab = np.rint(img_lab).astype(np.uint8)
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


    with torch.no_grad():
        for l, ab in test_data:
            l = l.to(device)
            ab = ab.to(device)
            model_ab = model(l).squeeze()

            l = l.squeeze(0)
            ab = ab.squeeze(0)


            l = l_un_normalize_transform(l)
            model_ab = ab_un_normalize_transform(model_ab)
            ab = ab_un_normalize_transform(ab)

            # img_2 = torch.cat([l, G_ab], dim=0)
            # img_2 = img_2.permute(1, 2, 0).numpy() * 255
            # img_2 = np.rint(img_2).astype(np.uint8)
            # img_2[:, :, [1, 2]] = 128
            # img_2 = cv2.cvtColor(img_2, cv2.COLOR_LAB2BGR)
            # cv2.imshow("img_gray", img_2)

    #         print(l.size())
    #         print(ab.size())
    #         print(G_ab.size())

            imgs = []
            # 预测图片
            img = concat_l_ab(l, model_ab)
            imgs.append(img)
            # 标准图片
            img = concat_l_ab(l, ab)
            imgs.append(img)

            # 输出图片，figsize为最终输出的大小
            _, axes = plt.subplots(1, 2, figsize=(32, 32))
            for ax, img in zip(axes.flatten(), imgs):
                ax.imshow(img)
                ax.axis("off")
                
test(10,151 , 1)