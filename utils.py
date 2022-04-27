import numpy as np
import pandas as pd
import plotly.express as px


def compute_rmse(y, ypred, ystd=1.):
    return np.mean((y - ypred) ** 2) ** 0.5 * ystd


def get_cv_idx(n, test_size=0.2, n_splits=2):
    train_idx, test_idx = [], []
    for _ in range(n_splits):
        idx = np.random.permutation(n)
        train_size = int(n * (1 - test_size)) if isinstance(test_size, float) else n - test_size
        train_idx.append(idx[:train_size])
        test_idx.append(idx[train_size:])
    return train_idx, test_idx


def load_csv(path, nrows=None):
    if nrows:
        return pd.read_csv(path, nrows=nrows)
    return pd.read_csv(path)


def save_submission_csv(path, df, values):
    print('Exporting results...')
    submission_df = pd.DataFrame(data={'index': df['index'].values,
                                       'fact_temperature': values.squeeze()})

    # Save the predictions into a csv file
    # Notice that this file should be saved under the directory `/kaggle/working`
    # so that you can download it later
    submission_df.to_csv(path, index=False)


def geo_plot(df, col_name):
    fig = px.scatter_geo(df, lat='fact_latitude', lon='fact_longitude', color=col_name)
    # fig = px.scatter(df, y='fact_latitude', x='fact_longitude', color=col_name, render_mode='webgl')
    fig.write_image(f"fig_{col_name}.jpeg")
    fig.show()
