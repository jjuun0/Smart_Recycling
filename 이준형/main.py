import train
import test
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import os
import PIL
import torch
import time

MODELNAME = 'model/transparent'

if __name__ == '__main__':
    random_seed = 42

    # train_data_path = 'archive/Garbage classification/Garbage classification'  # 학습에 사용한 데이터
    # train_data_path = 'D:/dataset/SAMSUNG/Desktop/dataset/plastic'  # 학습에 사용한 데이터
    train_data_path = 'C:/Users/SAMSUNG/Desktop/transparent'  # 학습에 사용한 데이터


    train_transformations = transforms.Compose([transforms.Resize((256, 256)),
                                                # transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                          # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                          ])

    train_dataset = ImageFolder(train_data_path, transform=train_transformations)
    classes = os.listdir(train_data_path)
    print(classes)

    total = len(train_dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size
    # train_ds, val_ds, test_ds = random_split(dataset, [train_size, 279, 279])
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    # ----- train -----
    # train.start_train(classes, train_ds, val_ds, batch_size=16, epochs=5)
    # test.test_folder(dataset, classes, test_ds)
    test_transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # ----- test folder -----
    # my_data = 'mydata'  # 테스트를 위한 데이터
    # my_dataset = ImageFolder(my_data, transform=test_transformations)
    # test.test_folder(my_dataset, classes, my_dataset)

    test_data = 'D:/dataset/test'
    test_dataset = ImageFolder(test_data, transform=test_transformations)
    test.test_folder(test_dataset, classes, test_dataset)

    print(f'model name : {MODELNAME}')

    # # ----- test file -----
    # # # transformation 을 하기 위해 PIL 로 이미지를 연다
    # img = PIL.Image.open('temp_img/plastic/2.jpeg')
    # # # img = PIL.Image.open('mydata/glass/glass12.jpg')
    # # # # img.show()
    # img_t = test_transformations(img)
    # # # print(img_t.size())
    # # # tf = transforms.ToPILImage()
    # # # img_p = tf(img_t)
    # # # img_p.show()
    # device = torch.device('cpu')
    # # # device = torch.device('cuda')
    # start = time.time()
    # print(test.test_image(device, classes, img_t))
    # print('predict time : ', time.time()-start)
