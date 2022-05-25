from pathlib import Path

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torchsummary import torchsummary

from conv_model import STgramMFN
from lab2.baseline.test import test
from lab2.baseline.model import VAE, Task2VAELightning, PaperConvModel
from lab2.baseline.util import select_dirs, file_list_generator


def train(params):
    # create working directory
    Path(params.model_directory).mkdir(exist_ok=True, parents=True)

    # test directories
    dirs = select_dirs(param=params, mode=params.mode)
    print(dirs)

    # fix random seeds
    torch.manual_seed(420)

    # TODO check what this does
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for target_dir in dirs:
        target = str(target_dir).split('/')[-1]
        print(f'==== Start training [{target}] with {torch.cuda.device_count()} GPU(s). ====')

        files = file_list_generator(target_dir)

        model = VAE(x_dim=params.VAE.x_dim, h_dim=params.VAE.h_dim, z_dim=params.VAE.z_dim).to(device)

        torchsummary.summary(model.to(device), input_size=(1, 640))

        task2 = Task2VAELightning(device, model, params, files, normalize=True)
        trainer = pl.Trainer(max_epochs=params.fit.epochs,
                             gpus=torch.cuda.device_count())
        trainer.fit(task2)

        model_file = f'{params.model_directory}/model_{target}.pth'
        torch.save(task2.model.state_dict(), model_file)
        print(f'saved {model_file}.\n')


def train_conv(params):
    # create working directory
    Path(params.model_directory).mkdir(exist_ok=True, parents=True)

    # test directories
    dirs = select_dirs(param=params, mode=params.mode)

    # fix random seeds
    torch.manual_seed(420)

    # TODO check what this does
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for target_dir in dirs:
        target = str(target_dir).split('/')[-1]
        print(f'==== Start training [{target}] with {torch.cuda.device_count()} GPU(s). ====')

        files = file_list_generator(target_dir)

        model = STgramMFN(num_class=41,
                          c_dim=params.feature.n_mels,
                          win_len=params.feature.win_length,
                          hop_len=params.feature.hop_length,
                          arcface=None)

        torchsummary.summary(model.to(device), input_size=(1, 640))

        task2 = PaperConvModel(device, model, params, files, normalize=True)
        trainer = pl.Trainer(max_epochs=params.fit.epochs,
                             gpus=torch.cuda.device_count())
        trainer.fit(task2)

        model_file = f'{params.model_directory}/model_2_{target}.pth'
        torch.save(task2.model.state_dict(), model_file)
        print(f'saved {model_file}.\n')


def main():
    params = {
        # inout directory
        "dev_directory": "data/dev_data/dev_data",
        "eval_directory": "data/eval_data/eval_data",
        "model_directory": "./model",
        "result_directory": "./result",
        "result_file": "result.csv",
        "max_fpr": 0.1,
        "mode": False,  # mode=True for development dataset, mode=False for evaluation dataset

        # preprocessing for mel-spectrogram
        "feature": {
            "n_mels": 128,
            "frames": 5,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 512,
            "power": 2.0
        },

        # training
        "fit": {
            "lr": 0.001,
            "b1": 0.9,
            "b2": 0.999,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 1000,
            "shuffle": True,
            "validation_split": 0.1,
            "verbose": 1},

        # model architecture
        "VAE": {
            "x_dim": 640,
            "h_dim": 400,
            "z_dim": 20}

    }
    params = EasyDict(params)
    # check_dataset('data/dev_data/dev_data')
    # check_dataset('data/eval_data/eval_data')
    # enable for baseline training
    # train(params)

    train_conv(params)
    test(params)


if __name__ == '__main__':
    main()
