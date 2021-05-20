import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils
import train
import random
from model import ResNet
import os
import DeviceDataLoader

def predict_image(dataset, img, model):
    device = train.get_default_device()
    # Convert to a batch of 1
    xb = train.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label


    return dataset.classes[preds[0].item()]

def predict_image_test(device, classes, img):
    model = ResNet(classes)
    model.load_state_dict(torch.load('./' + train.MODELNAME, map_location=device))
    model.eval()
    # model.cuda()

    # device = DeviceDataLoader.get_default_device()
    
    # Convert to a batch of 1
    xb = train.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # print('predict: ', yb[0])
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)

    # Retrieve the class label

    return prob[0].item(), classes[preds[0].item()]
    # return dataset.classes[preds[0].item()]



#
def test(dataset, classes, test_ds):
    # classes = os.listdir(dataset)
    model = ResNet(classes)
    model.load_state_dict(torch.load('./' + train.MODELNAME))
    model.eval()
    model.cuda()

    count = 0
    # for i in range(279):
    for i in range(len(test_ds)):
        # n = random.randint(0, 279)
        img, label = test_ds[i]
        # plt.imshow(img.permute(1, 2, 0))
        # plt.show()
        original = dataset.classes[label]
        predict = predict_image(dataset, img, model)
        # print('Label:', dataset.classes[label], ', Predicted:', predict_image(dataset, img, model))
        if original != predict:
            # plt.imshow(img.permute(1, 2, 0))
            # plt.show()
            print('Label:', dataset.classes[label], ', Predicted:', predict_image(dataset, img, model))
            count += 1

    print('error : ', count)
    print(count/len(test_ds))

if __name__ == '__main__':
    predict_image(data, img, model)