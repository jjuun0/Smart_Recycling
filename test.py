import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils
import main
import train
import random
from model import ResNet
import os
import DeviceDataLoader
from PIL import Image


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


def test_image(device, classes, img):
    """ model predicts an image """
    model = ResNet(classes)
    model.load_state_dict(torch.load('./' + main.MODELNAME, map_location=device))
    model.eval()
    model.cuda()

    # Convert to a batch of 1
    xb = train.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    print(yb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)

    # Retrieve the class label
    return prob[0].item(), classes[preds[0].item()]


def predict(device, model, classes, img):
    xb = DeviceDataLoader.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)

    # Retrieve the class label
    return prob[0].item(), classes[preds[0].item()]


def predict_cuda(img):
    device = torch.device('cuda')
    classes = ['glass', 'plastic', 'metal']
    trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # https://discuss.pytorch.org/t/typeerror-pic-should-be-pil-image-or-ndarray-got-class-numpy-ndarray/20134/4
    img = Image.fromarray(img)
    img = trans(img)

    model = ResNet(classes)
    model.load_state_dict(torch.load('./' + main.MODELNAME, map_location=device))
    model.eval()
    model.cuda()

    # Convert to a batch of 1
    xb = train.to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)

    # Retrieve the class label
    return prob[0].item(), classes[preds[0].item()]


def test_folder(dataset, classes, test_ds):
    """ model predicts images in folder """
    model = ResNet(classes)
    model.load_state_dict(torch.load('./' + main.MODELNAME))
    model.eval()
    model.cuda()

    count = 0
    # for i in range(279):
    for i in range(len(test_ds)):
        img, label = test_ds[i]
        # plt.imshow(img.permute(1, 2, 0))
        # plt.show()
        original = dataset.classes[label]
        predict = predict_image(dataset, img, model)
        if original != predict:
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
            print('Label:', dataset.classes[label], ', Predicted:', predict_image(dataset, img, model))
            count += 1

    print('error : ', count)
    print(count/len(test_ds))


if __name__ == '__main__':
    predict_image(data, img, model)