import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms



class GarbageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = os.listdir(root_dir)

        imgs = []
        labels_idx = []

        for idx, folder in enumerate(self.classes):
            folder_path = os.path.join(self.root_dir, folder)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                imgs.append(img_path)  # getitem 에서 image open
                labels_idx.append(idx)

        data = [(x, y) for x, y in zip(imgs, labels_idx)]
        self.data = data

    def __getitem__(self, idx):
        img_path, label_idx = self.data[idx]

        img = Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        return img, label_idx

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    train_transformations = transforms.Compose([transforms.Resize((256, 256)),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
    dataset = GarbageDataset('D:/dataset/Garbage classification', transforms=train_transformations)
    first = dataset[0]

    print(os.listdir('D:/dataset/Garbage classification'))
