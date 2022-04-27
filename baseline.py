import time

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


def drop_irrelevant_features(df):
    # geo_plot(df[col_selection], col_name='fact_temperature')
    X = df[COL_SELECTION].dropna().iloc[:, 1:-1].values
    y = df[COL_SELECTION].dropna().iloc[:, -1].values
    return X, y


def main():
    df = load_csv(f'{DATA_DIR}/public/train.csv')
    X, y = drop_irrelevant_features(df)

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
    x_val = df_test[COL_SELECTION[:-1]].iloc[:, 1:].values

    y_pred = model.predict(x_val)
    save_submission_csv(f'{RESULT_DIR}/{time.time()}_submission.csv', df_test, y_pred)


if __name__ == '__main__':
    main()
