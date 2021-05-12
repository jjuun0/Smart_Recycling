import train
import test
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import os

if __name__ == '__main__':
    random_seed = 42

    data_dir = 'archive/Garbage classification/Garbage classification'  # 학습에 사용한 데이터
    my_data = 'mydata'  # 테스트를 위한 데이터
    # loading and splitting data
    # transformations
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    # dataset = ImageFolder(data_dir, transform=transformations)
    my_dataset = ImageFolder(my_data, transform=transformations)
    classes = os.listdir(data_dir)
    print(classes)

    # train_ds, val_ds, test_ds = random_split(dataset, [835, 279, 279])
    # print(len(train_ds), len(val_ds), len(test_ds))
    # train.start_train(data_dir, classes, train_ds, val_ds)
    # test.test(dataset, classes, test_ds)
    test.test(my_dataset, classes, my_dataset)
