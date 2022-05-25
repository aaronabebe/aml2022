import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from lab2.baseline.model import VAE
from lab2.baseline.pytorch_common import load_weights, get_machine_id_list_for_test, test_file_list_generator, save_csv
from lab2.baseline.util import select_dirs, file_to_vector_array, normalize_0to1, ToTensor1ch


def test(params):
    os.makedirs('../submission', exist_ok=True)

    # load base directory
    dirs = select_dirs(param=params, mode=params.mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # PyTorch version specific...
    to_tensor = ToTensor1ch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.pth".format(model=params.model_directory,
                                                               machine_type=machine_type)

        # load model file
        if not os.path.exists(model_file):
            print("{} model not found ".format(machine_type))
            sys.exit(-1)
        print("loading model: {}".format(model_file))
        model = VAE(x_dim=params.VAE.x_dim, h_dim=params.VAE.h_dim, z_dim=params.VAE.z_dim).to(device)
        load_weights(model, model_file)
        # torchsummary.summary(model, params.VAE.x_dim)
        model.eval()

        if params.mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_str, mode=params.mode)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                result=params.result_directory,
                machine_type=machine_type,
                id_str=id_str)
            anomaly_score_list = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                data = file_to_vector_array(file_path,
                                            n_mels=params.feature.n_mels,
                                            frames=params.feature.frames,
                                            n_fft=params.feature.n_fft,
                                            hop_length=params.feature.hop_length,
                                            power=params.feature.power)
                data = normalize_0to1(data)
                with torch.no_grad():
                    yhat = model(to_tensor(data)).cpu().detach().numpy().reshape(data.shape)
                    errors = np.mean(np.square(data - yhat), axis=1)
                y_pred[file_idx] = np.mean(errors)
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            print("anomaly score result ->  {}".format(anomaly_score_csv))

            if params.mode:
                # append AUC and pAUC to lists
                auc = roc_auc_score(y_true, y_pred)
                p_auc = roc_auc_score(y_true, y_pred, max_fpr=params.max_fpr)
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                print("AUC : {}".format(auc))
                print("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if params.mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if params.mode:
        # output results
        result_path = "{result}/{file_name}".format(result=params.result_directory, file_name=params.result_file)
        print("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)


    df1 = pd.read_csv('../result/anomaly_score_slider_id_01.csv', delimiter=',', header=None)
    df3 = pd.read_csv('../result/anomaly_score_slider_id_03.csv', delimiter=',', header=None)
    df5 = pd.read_csv('../result/anomaly_score_slider_id_05.csv', delimiter=',', header=None)

    dfs = [df1, df3, df5]
    eval_df = pd.concat(dfs)
    eval_df = eval_df.rename(columns={0: "file_name", 1: "anomaly_score"})

    eval_df.to_csv('./submission/submission.csv', index=False)
