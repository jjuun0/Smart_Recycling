from model import ResNet
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import train
from torch.utils.tensorboard import SummaryWriter
import copy

# writer = SummaryWriter('runs/' + train.MODELNAME + '/test_result')

def predict(model_path, classes, test_dir):
    test_transformations = transforms.Compose([transforms.Resize((256, 256)),
                                                # transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()
                                                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
    test_dataset = ImageFolder(test_dir, transform=test_transformations)

    device = torch.device('cuda:0')
    model = ResNet(classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    images_total_num = {i: 0 for i in classes}
    images_success_num = {i: 0 for i in classes}
    images_error_num = {i: 0 for i in classes}
    images_error_class = {}
    images_success_confidence = {i: 0 for i in classes}

    for class_name in classes:
        temp = copy.deepcopy(classes)
        temp.remove(class_name)
        images_error_class[class_name] = {t: 0 for t in temp}


    with torch.no_grad():
        for i in range(len(test_dataset)):
            input, label = test_dataset[i]
            input = input.unsqueeze(0).to(device)
            origin_label = test_dataset.classes[label]
            images_total_num[origin_label] += 1

            output = model(input)
            prob, pred = torch.max(output, 1)

            pred_label = classes[pred[0].item()]

            if origin_label != pred_label:
                # print(f'origin: {origin_label} / pred: {pred_label}')
                images_error_num[origin_label] += 1
                images_error_class[origin_label][pred_label] += 1
            else:
                images_success_confidence[origin_label] += float(f'{prob[0].item()*100}')
                images_success_num[origin_label] += 1

        for class_name in images_total_num.keys():
            print(f'---------- {class_name} ----------')
            if images_success_num[class_name] > 0:
                average_confidence = f'{images_success_confidence[class_name]/images_success_num[class_name]:.1f}'
                print(f'prediction average success confidence: {average_confidence}%')

            if images_total_num[class_name] > 0:
                failure_rate = f'{images_error_num[class_name]/images_total_num[class_name]*100:.1f}'
                print(f'prediction failure rate: {failure_rate}%')

            for failure_class, num in images_error_class[class_name].items():
                if num > 0:
                    failure_class_rate = f'{num / images_error_num[class_name]*100:.1f}'
                else:
                    failure_class_rate = 0
                print(f'----- {failure_class}: {failure_class_rate}%')

            print()


if __name__ == '__main__':
    classes = ['glass', 'metal', 'plastic']
    test_dir = 'D:/dataset/test'
    device = torch.device('cuda:0')
    model = ResNet(classes).to(device)
    model.load_state_dict(torch.load('./' + train.MODELNAME, map_location=device))
    predict(test_dir, device, model)