import csv
import glob
import itertools
import os
import re

import numpy
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torchsummary


def load_weights(model, weight_file):
    model.load_state_dict(torch.load(weight_file))


def summary(device, model, input_size=(1, 640)):
    torchsummary.summary(model.to(device), input_size=input_size)


def summarize_weights(model):
    summary = pd.DataFrame()
    for k, p in model.state_dict().items():
        p = p.cpu().numpy()
        df = pd.Series(p.ravel()).describe()
        summary.loc[k, 'mean'] = df['mean']
        summary.loc[k, 'std'] = df['std']
        summary.loc[k, 'min'] = df['min']
        summary.loc[k, 'max'] = df['max']
    return summary


def show_some_predictions(dl, model, start_index, n_samples, image=False):
    shape = (-1, 64, 64) if image else (-1, 640)
    x, y = next(iter(dl))
    with torch.no_grad():
        yhat = model(x)
    x = x.cpu().numpy().reshape(shape)
    yhat = yhat.cpu().numpy().reshape(shape)
    print(x.shape, yhat.shape)
    for sample_idx in range(start_index, start_index + n_samples):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if image:
            axs[0].imshow(x[sample_idx])
            axs[1].imshow(yhat[sample_idx])
        else:
            axs[0].plot(x[sample_idx])
            axs[1].plot(yhat[sample_idx])


class ToTensor1ch(object):
    """PyTorch basic transform to convert np array to torch.Tensor.
    Args:
        array: (dim,) or (batch, dims) feature array.
    """

    def __init__(self, device=None, image=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.non_batch_shape_len = 2 if image else 1

    def __call__(self, array):
        # (dims)
        if len(array.shape) == self.non_batch_shape_len:
            return torch.Tensor(array).unsqueeze(0).to(self.device)
        # (batch, dims)
        return torch.Tensor(array).unsqueeze(1).to(self.device)


def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav",
                             mode=True):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files
    mode : bool (default=True)
        True/False used for development/evaluation dataset

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    print("target_dir : {}".format(target_dir + "_" + id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        print("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n=========================================")

    return files, labels
