import train
import test
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import os
import PIL
import torch

MODELNAME = 'nomalized_resnet50_4'

if __name__ == '__main__':
    random_seed = 42

    data_dir = 'archive/Garbage classification/Garbage classification'  # 학습에 사용한 데이터
    my_data = 'mydata'  # 테스트를 위한 데이터
    # loading and splitting data
    # transformations
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(data_dir, transform=transformations)
    classes = os.listdir(data_dir)
    print(classes)

    train_ds, val_ds, test_ds = random_split(dataset, [835, 279, 279])


    # train.start_train(data_dir, classes, train_ds, val_ds)
    # test.test(dataset, classes, test_ds)
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    my_dataset = ImageFolder(my_data, transform=trans)
    test.test(my_dataset, classes, my_dataset)

    # img = PIL.Image.open('temp_img/plastic2.jpeg')
    # # img.show()

    # img_t = trans(img)
    # print(img_t.size())
    #
    # tf = transforms.ToPILImage()
    # img_p = tf(img_t)
    # img_p.show()
    #
    # device = torch.device('cpu')
    # print(test.predict_image_test(device, classes, img_t))
