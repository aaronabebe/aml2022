import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import get_cv_idx, load_csv, save_submission_csv

DATA_DIR = './data'
RESULT_DIR = './results'

COL_SELECTION = [
    'index', 'fact_time', 'fact_latitude', 'fact_longitude',
    'topography_bathymetry', 'sun_elevation', 'cmc_precipitations',
    'gfs_a_vorticity', 'gfs_cloudness', 'gfs_clouds_sea', 'gfs_humidity',
    'fact_temperature'
]

BASE_SELECTION = [
    'index', 'fact_time', 'fact_latitude', 'fact_longitude'
]


def grid_search(model, X, y):
    train_idx, cv_idx = get_cv_idx(len(X), test_size=0.3, n_splits=10)

    param_grid = {
        'lasso__alpha': [0.001, 0.01, 0.1, 1, 10],
    }

    search = GridSearchCV(
        model,
        param_grid,
        n_jobs=-1,
        verbose=1,
        cv=zip(train_idx, cv_idx),
        scoring='neg_root_mean_squared_error'
    ).fit(X, y)
    print('Best parameters set found on cv set:')
    print(search.best_params_)
    print()
    print('Grid scores on cv set:')
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std, params))
    return search


def drop_irrelevant_features(df, selection=False):
    # geo_plot(df[col_selection], col_name='fact_temperature')
    if selection:
        X = df[selection].dropna().iloc[:, 1:-1].values
        y = df[selection].dropna().iloc[:, -1].values
    else:
        X = df.dropna().iloc[:, 1:-1].values
        y = df.dropna().iloc[:, -1].values
    return X, y


def co_variance_analysis(df, top=30, plot=False):
    cor = df[df.columns[1:]].corr()["fact_temperature"]
    df_plot = pd.DataFrame({'feature': cor.index.values, 'correlation': cor.values})
    df_plot_top = df_plot.sort_values(by='correlation', ascending=False)[1:top]
    if plot:
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Scatter(x=df_plot['feature'], y=df_plot['correlation'], text=f'Top {top} Correlation Values'),
            row=1, col=1
        )

        # remove first entry which is fact_temperature itself
        fig.add_trace(
            go.Scatter(x=df_plot_top['feature'], y=df_plot_top['correlation'], text=f'Top {top} Correlation Values'),
            row=1, col=2
        )
        fig.show()
    return df_plot_top


def feature_preprocessing(df):
    df = df[(df['cmc_available'] == 1.0) & (df['gfs_available'] == 1.0) & (df['wrf_available'] == 1.0)]
    df['geographical_zone'] = [0 if x > 66.5 else 1 if x < -66.5 else 2 if x > 23.5 else 3 if x < 23.5 else 4 for x in
                               df['fact_latitude']]
    df['cmc_0_0_7_1000'] = df['cmc_0_0_7_1000'] * 2
    df['cmc_0_0_7'] = df[
        ['cmc_0_0_7_1000', 'cmc_0_0_7_2', 'cmc_0_0_7_500', 'cmc_0_0_7_700', 'cmc_0_0_7_850', 'cmc_0_0_7_925']].mean(
        axis=1)
    df['gfs_cloudness_buckets'] = [0 if x == 0 else 1 if x <= 1 else 2 if x <= 2 else 3 for x in df['gfs_cloudness']]
    return df


def main():
    df = load_csv(f'{DATA_DIR}/public/train.csv')

    # corr_features = co_variance_analysis(df)
    # corr_features = corr_features.loc[corr_features['correlation'] >= 0.7]
    # geo_plot(df[PLOT_SELECTION], 'fact_temperature')

    selection = [
        'index', 'fact_time', 'fact_latitude', 'fact_longitude',
        'sun_elevation', 'cmc_precipitations', 'wrf_t2_interpolated',
        'gfs_temperature_95000', 'gfs_temperature_90000', 'gfs_temperature_65000',
        'gfs_temperature_55000', 'gfs_temperature_45000', 'gfs_temperature_35000',
        'gfs_a_vorticity', 'gfs_humidity', 'climate_temperature', 'gfs_precipitable_water',
        'cmc_0_1_0_0', 'cmc_0_0_6_2', 'cmc_0_0_7', 'gfs_cloudness_buckets', 'geographical_zone',
        'fact_temperature',
    ]
    df = feature_preprocessing(df)
    X, y = drop_irrelevant_features(df, selection)

    Xtr, Xval, ytr, yval = train_test_split(X, y, random_state=1, test_size=0.40)
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_jobs=8, n_estimators=24, verbose=True))

    model.fit(Xtr, ytr)
    ypred_tr = model.predict(Xtr)
    ypred_val = model.predict(Xval)

    print(f'Train RMSE: {mean_squared_error(ytr, ypred_tr, squared=False):.3f}')
    print(f'Valid RMSE: {mean_squared_error(yval, ypred_val, squared=False):.3f}')

    # NOTE: uncomment for gridsearch
    # search = grid_search(model, Xtr, ytr)
    # ypred_val = search.predict(Xval)
    # print('Error on the validation set')
    # print(f'Valid RMSE: {mean_squared_error(yval, ypred_val, squared=False):.3f}')

    df_test = load_csv(f'{DATA_DIR}/public/test_feat.csv')
    df_test = feature_preprocessing(df_test)
    x_val = df_test[selection[:-1]].iloc[:, 1:].values

    y_pred = model.predict(x_val)
    save_submission_csv(f'{RESULT_DIR}/{time.time()}_submission.csv', df_test, y_pred)


if __name__ == '__main__':
    main()
