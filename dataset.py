# datasets.py
from torch.utils.data import Dataset
import torchvision.datasets
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

class LabImageDataset(Dataset):
    """
    将一个 :class:`VisionDataset` 类型的数据集转换至Lab空间\n
    数据集的输入必须是以 :class:`PIL.Image.Image` 或者 :class:`numpy.ndarray` 作为input的另一数据集\n
    数据集输出的input为L通道的tensor对象\n
    output为ab通道的tensor对象（不再是3通道图片）
    """
    def __init__(self, image_dataset: VisionDataset, image_transform=None):
        """
        Args:
            image_dataset (Dataset): 一个pytorch的图片Dataset.
            image_transform (callable, optional): 可选的Transform
        """
        self.image_dataset = image_dataset
        if image_transform is None:
            self.image_transform = transforms.ToTensor()
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor(), image_transform])

    def __len__(self):
        return self.image_dataset.__len__()

    def __getitem__(self, idx):
        img, _ = self.image_dataset.__getitem__(idx)
        # PIL图片转换成numpy数组
        if isinstance(img, Image.Image):
            img = np.array(img)
        # 转换为lab色彩空间
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # 转换为tensor，并进行处理
        img = self.image_transform(img)
        # 分离出l与ab通道
        channel_l = img[0:1, ...]
        channel_ab = img[1:3, ...]

        return channel_l, channel_ab

class ColorizationDataset(Dataset):
    """
    将文件夹内的图片加载为数据集
    """
    def __init__(self, image_root, image_transform=None, limit=None, seed=123):
        """
        Args:
            image_root (str): 只包含图片的文件夹路径.
            image_transform (callable, optional): 可选的Transform
            limit (int): 加载的最大数量限制
            seed (int): 抽取的随机种子
        """
        self.limit = limit
        self.image_paths = []

        import glob
        self.paths = glob.glob(image_root + "/*.jpg") # Grabbing all the image file names

        if limit is not None:
            np.random.seed(seed)
            self.paths = np.random.choice(self.paths, limit, replace=False)

        if image_transform is None:
            self.image_transform = transforms.ToTensor()
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor(), image_transform])

    def __getitem__(self, index):
        image_path = self.paths[index]
        img = cv2.imread(image_path)

        # 转换为lab色彩空间
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # 转换为tensor，并进行处理
        img = self.image_transform(img)

        # 分离出l与ab通道
        channel_l = img[0:1, ...]
        channel_ab = img[1:3, ...]

        return channel_l, channel_ab

    def __len__(self):
        return len(self.paths)