import torch
from model import ResNet
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from DeviceDataLoader import DeviceDataLoader, get_default_device, to_device
import main
import torchvision.models as models
import torch.nn as nn


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            
            loss = model.training_step(batch)
            train_losses.append(loss)
            
            loss.backward()
            optimizer.step()
            
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def start_train(classes, train_ds, val_ds, batch_size, epochs):
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)

    model = ResNet(classes)

    device = get_default_device()
    print(device)

    to_device(model, device)

    # epochs = 8
    opt_func = torch.optim.Adam
    lr = 5.5e-5

    history = fit(epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)
    plot_losses(history)

    torch.save(model.state_dict(), "./" + main.MODELNAME, _use_new_zipfile_serialization=False)

