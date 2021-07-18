from model import ResNet
import torch
from CustomDataset import GarbageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

# MODELNAME = 'model/model1'
# writer = SummaryWriter('runs/' + MODELNAME.split('/')[-1])

def train(modelname, train_dir, epochs=6):
    writer = SummaryWriter('runs/' + modelname)

    train_transformations = transforms.Compose([transforms.Resize((256, 256)),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
    dataset = GarbageDataset(train_dir, transforms=train_transformations)

    total = len(dataset)
    train_size = int(0.9 * total)
    val_size = total - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True, num_workers=0)

    # train_dataset_size = {x: len(train_dataset[x]) for x in ['train', 'val']}

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    model = ResNet(dataset.classes).to(device)

    dataiter = iter(train_dataloader)
    temp_images, temp_labels = dataiter.next()
    temp_images = temp_images.to(device)
    writer.add_graph(model, temp_images)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5.5e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7)

    for epoch in range(epochs):
        print(f'Epoch {epoch} / {epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects.double() / train_size

            else:
                epoch_loss = running_loss / val_size
                epoch_acc = running_corrects.double() / val_size

            writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} acc', epoch_acc, epoch)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    torch.save(best_model_wts, "./model/" + modelname, _use_new_zipfile_serialization=False)
    writer.close()

if __name__ == '__main__':
    garbage_train_dir = 'D:/dataset/Garbage classification'
    train(garbage_train_dir)


# tensorboard --logdir ./runs/model1 --port 6008