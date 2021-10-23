import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import albumentations as alm


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image_ccf/*.png'))  # 指定训练图片位置：image / image_ccf

    def augment(self, img, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        # 1.OpenCV
        flip = cv2.flip(img, flipCode)
        # 2.albumentations图像增强库
        # if flipCode == 1:
        #     flip = alm.HorizontalFlip(p=1)(image=img)['image']
        # elif flipCode == 0:
        #     flip = alm.VerticalFlip(p=1)(image=img)['image']
        # elif flipCode == -1:
        #     flip = alm.HorizontalFlip(p=1)(image=img)['image']
        #     flip = alm.VerticalFlip(p=1)(image=flip)['image']
        # else:
        #     flip = img
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image_ccf', 'label_ccf')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # print('read image.shape: ', image.shape)
        # print('read label.shape: ', label.shape)

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # print('image.shape: ', image.shape)
        # print('label.shape: ', label.shape)
        # if label.shape == (1080, 1920):
        #     while True:
        #         print('label.path: ', label_path)
        # image 和 label分辨率匹配处理（不一致，则resize为相同，否则无法训练！）
        if image.shape != label.shape:
            image = cv2.resize(image, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_AREA)

        # print('resize image.shape: ', image.shape)
        # print('resize label.shape: ', label.shape)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255

        # 随机进行数据增强，为2时不做处理
        # print('image.shape: ', image.shape)
        # print('label.shape: ', label.shape)
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("D:/Jiangyu/CV/Code/DeepLearning/U-Net/data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=False)
    index = 0
    for image, label in train_loader:
        index += 1
        print('img{}: {}'.format(index, image.shape))
