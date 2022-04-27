import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


DATA_DIR = './data'


def compute_rmse(y, ypred, ystd=1.):
    return np.mean((y - ypred)**2)**0.5 * ystd

def get_cv_idx(n, test_size=0.2, n_splits=2):
    train_idx, test_idx = [], []
    for _ in range(n_splits):
        idx = np.random.permutation(n)
        train_size = int(n * (1 - test_size)) if isinstance(test_size, float) else n - test_size
        train_idx.append(idx[:train_size])
        test_idx.append(idx[train_size:])
    return train_idx, test_idx


df = pd.read_csv(f'{DATA_DIR}/public/train.csv')

print(df.head())

col_selection = ['index', 'fact_time', 'fact_latitude', 'fact_longitude', 'fact_temperature']

X = df[col_selection].dropna().iloc[:, 1:-1].values
y = df[col_selection].dropna().iloc[:, -1].values

Xmean, Xstd, ymean, ystd = X.mean(0), X.std(0), y.mean(), y.std()
X = (X - Xmean) / Xstd
y = (y - ymean) / ystd


Xtr, Xval, ytr, yval = train_test_split(X, y, random_state=1, test_size=3000)

model = Lasso(alpha=1)
model.fit(Xtr, ytr)
ypred_tr = model.predict(Xtr)
ypred_val = model.predict(Xval)

print(f'Train RMSE: {compute_rmse(ytr, ypred_tr, ystd):.3f}')
print(f'Valid RMSE: {compute_rmse(yval, ypred_val, ystd):.3f}')


print('\ndoing gridsearch...')
train_idx, cv_idx = get_cv_idx(len(Xtr), test_size=10000, n_splits=10)


param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]
}

search = GridSearchCV(model,
                      param_grid,
                      n_jobs=-1,
                      verbose=1,
                      cv=zip(train_idx, cv_idx),
                      scoring='neg_root_mean_squared_error').fit(Xtr, ytr)
print('Done!')

print("Best parameters set found on cv set:")
print(search.best_params_)
print()
print("Grid scores on cv set:")
means = search.cv_results_["mean_test_score"]
stds = search.cv_results_["std_test_score"]
for mean, std, params in zip(means, stds, search.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (-mean * ystd, (std * ystd) * 2, params))
print()
print("Error on the validation set")
ypred_val = search.predict(Xval)
print(f'Valid RMSE: {compute_rmse(yval, ypred_val, ystd):.3f}')

