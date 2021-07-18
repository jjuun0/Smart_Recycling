import torch.nn as nn
import torchvision.models as models
import torch


class ResNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(classes))

    def forward(self, xb):
        # return torch.sigmoid(self.network(xb))
        # return self.network(xb)
        return torch.softmax(self.network(xb), 1)


if __name__ == '__main__':
    classes = ['1', '2', '3']
    model = ResNet(classes)
    print(model)