# train_utils.py
import time
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from dataset import ColorizationDataset
from model3 import CNNGenerator

def init_weights(mean, std):
    def init(m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                m.bias.data.zero_()

    return init

def load_data(dataset, batch_size=1, shuffle=True, subset_indices=None, num_workers=0, pin_memory=False):
    """载入指定的 :class:`torch.utils.data.dataset` """
    if subset_indices is not None:
        subset = Subset(dataset, subset_indices)
    else:
        subset = dataset
    return DataLoader(subset, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)

epochs = 250
batch_size = 256
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

train_set = ColorizationDataset('/root/autodl-tmp/coco3200/train2017', 
                                transforms.Compose([transforms.Normalize(mean=normalize_mean * 3, std=normalize_std * 3), 
                                                    transforms.Resize((image_size, image_size))]),
                                3000)  # 载入前3000张图片

train_data = load_data(train_set, batch_size=batch_size, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================模型====================

# 这是ResUnet的实现
# Generator 1输入(L通道)，2输出（AB通道）64层开卷
G = CNNGenerator(G_in_channels, G_out_channels)
# Discriminator 3通道，64层开卷


# ====================初始化====================
start_epoch = 0
G.apply(init_weights(mean=0.0, std=0.02))
#====================or继续====================
# start_epoch = 200
# G.load_state_dict(torch.load(r'/root/autodl-tmp/GAN/Model/netG_epoch_200.pth'))
# D.load_state_dict(torch.load(r'/root/autodl-tmp/GAN/Model/netD_epoch_200.pth'))


G: nn.Module = G.to(device)

G.train()

# 损失函数
BCELossWithSigmoid = nn.BCEWithLogitsLoss().to(device)
L1Loss = nn.L1Loss().to(device)

# 优化器
optG = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, beta2))
#继续训练记得一并载入优化器
# optG.load_state_dict(torch.load(r'/root/autodl-tmp/GAN/Model/optG_epoch_200.pth'))
# optD.load_state_dict(torch.load(r'/root/autodl-tmp/GAN/Model/optD_epoch_200.pth'))

# ====================训练====================
# 记录
writer = SummaryWriter("/root/autodl-tmp/CNN_logs_train")
from tqdm import tqdm


def train() :
    for epoch in range(start_epoch, epochs):
        G_losses = []

        print(f'epoch {epoch} training')
        
        # 训练
        G.train()
        for sample, truth in tqdm(train_data):
            # 启用GPU
            sample: torch.Tensor = sample.to(device)
            truth: torch.Tensor = truth.to(device)
                
            # ====================鉴别器D的优化====================
            # 论文中有说明，把输入x(sample)纳入鉴别器的计算，实际效果会更好
            # 最大化 log(D(x, y)) + log(1 - D(x, G(x, z)))
            # 值域为(-∞,0)，即还是使式子趋于0
            # ===================================================

            # 对于正确的输出，鉴别器D应该接近1（真）时误差

            # 对于伪造的输出，鉴别器D应该接近0（假）时误差最小
            G_result = G(sample)


            # ====================生成器G的优化====================
            # 最小化 log(1 - D(x, G(x, z)))
            # 即使上式趋于-∞，以扰乱判别器D
            # 该项是D_loss的第二项，相当于增加D的误差
            #
            # 但是优化器只能往损失函数趋于0的方向优化，因此使用基本等价的下式：
            # 最大化 log(D(x, G(x, z)))
            # 值域为(-∞,0)，即还是使式子趋于0
            # 对比D_loss的第一项log(D(x, y)
            # 这相当于D把G生成的图片G(x, z)误判成真实图片y
            # 借助G我们即可获得以假乱真的图片
            # ===================================================
            G.zero_grad()



            G_loss = L1_Lambda * L1Loss(G_result, truth)
            G_loss.backward()
            optG.step()

            # print(f'G loss: {G_loss.item()}')
            G_losses.append(G_loss.item())
        G_avg_loss = sum(G_losses) / len(G_losses)

        print(f'G loss(train) {G_avg_loss}')
        writer.add_scalar("train_G(train)", G_avg_loss, epoch)
        
        # 校验
    #     print(f'epoch {epoch} validation')
        
    #     D_losses = []
    #     G_losses = []
    #     G.eval()
    #     D.eval()
    #     with torch.no_grad():
    #         for sample, truth in tqdm(val_data):
    #             sample: torch.Tensor = sample.to(device)
    #             truth: torch.Tensor = truth.to(device)

    #             # D
    #             # 对于正确的输出，鉴别器D应该接近1（真）时误差最小
    #             D_pred = D(sample, truth).squeeze()
    #             D_real_loss = BCELossWithSigmoid(D_pred, torch.ones(D_pred.size()).to(device))

    #             # 对于伪造的输出，鉴别器D应该接近0（假）时误差最小
    #             G_result = G(sample)
    #             D_pred = D(sample, G_result.detach()).squeeze()  # G作为常量输入
    #             D_fake_loss = BCELossWithSigmoid(D_pred, torch.zeros(D_pred.size()).to(device))

    #             D_loss = (D_real_loss + D_fake_loss) * 0.5
    #             D_losses.append(D_loss.item())

    #             # G
    #             D_pred = D(sample, G_result).squeeze()  # G作为变量输入

    #             G_loss = BCELossWithSigmoid(D_pred, torch.ones(D_pred.size()).to(device)) + L1_Lambda * L1Loss(G_result, truth)
    #             G_losses.append(G_loss.item())
            
    #     G_avg_loss = sum(G_losses) / len(G_losses)
    #     D_avg_loss = sum(D_losses) / len(D_losses)

    #     print(f'G loss(val) {G_avg_loss}')
    #     print(f'D loss(val) {D_avg_loss}')
    #     writer.add_scalar("train_G(val)", G_avg_loss, epoch)
    #     writer.add_scalar("train_D(val)", D_avg_loss, epoch)

        # 保存模型
        epoch_h = epoch + 1
        if epoch_h > 0:
            torch.save(G.state_dict(), '/root/autodl-tmp/GAN/CNN_Model/netG_epoch_%d.pth' % epoch_h)

            
        # 最优保存
    #     if G_avg_loss < G_min_loss:
    #         print(f'loss in G:{G_avg_loss} is better than min loss {G_min_loss}')
    #         G_min_loss = G_avg_loss
    #         print(f'saving model as best...')
    #         torch.save(G.state_dict(), 'bestG_netG.pth')
    #         torch.save(D.state_dict(), 'bestG_netD.pth')


    try:
        # 保存优化器
        torch.save(optG.state_dict(), '/root/autodl-tmp/GAN/CNN_Model/optG_epoch_%d.pth' % epoch_h)
    
        time.sleep(4)
    except:
        print("did not even train, nothing will be saved")
        pass

train()