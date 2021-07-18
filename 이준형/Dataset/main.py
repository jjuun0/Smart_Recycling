from config_parser import ConfigParser
import train
import test
import os


if __name__ == '__main__':
    # config parsing & setting
    config = ConfigParser('./config.json')
    model_path = config['model_path']
    train_dataset_path = config['train_dataset_path']
    test_dataset_path = config['test_dataset_path']
    do_train = config['do_train']
    do_test = config['do_test']

    if do_train == 'True':
        print('***** Train Step *****')
        model = model_path.split('/')[-1]
        train.train(model, train_dataset_path, 8)

    if do_test == 'True':
        print('***** Test Step *****')
        classes = os.listdir(train_dataset_path)
        test.predict(model_path, classes, test_dataset_path)


# tensorboard --logdir ./runs/model1 --port 6008