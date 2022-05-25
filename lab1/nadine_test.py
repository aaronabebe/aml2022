import pandas as pd 
import time
from matplotlib import pyplot as plt

df = pd.read_csv('../input/eurecom-aml-2022-challenge-1/public/train.csv', low_memory=True)
df = df.sample(n = 1000)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def drop_irrelevant_features(df):
    # geo_plot(df[col_selection], col_name='fact_temperature')
    X = df[COL_SELECTION].dropna().iloc[:, 1:-1].values
    y = df[COL_SELECTION].dropna().iloc[:, -1].values
    return X, y

# Column Selection
COL_SELECTION = [
    'index', 'fact_time', 'fact_latitude', 'fact_longitude',
    'sun_elevation', 'cmc_precipitations',
    'gfs_a_vorticity', 'gfs_humidity',
    'fact_temperature', 'cmc_0_1_0_0', 'cmc_0_0_6_2', 'cmc_0_0_7', 'gfs_cloudness_buckets', 'geographical_zone'
]


#Some new Columns
df['geographical_zone'] = [ 0 if x > 66.5 else 1 if x < -66.5 else 2 if x > 23.5 else 3 if x < 23.5 else 4 for x in df['fact_latitude']] 
df['cmc_0_0_7'] = (df['cmc_0_0_7_1000'] * 2 + df['cmc_0_0_7_2'] + df['cmc_0_0_7_500'] + df['cmc_0_0_7_700'] + df['cmc_0_0_7_850'] + df['cmc_0_0_7_925']) / 7
df['gfs_cloudness_buckets'] = [ 0 if x == 0 else 1 if x <= 1  else 2 if x <= 2 else 3 for x in df['gfs_cloudness']] 
df.head()   

X, y = drop_irrelevant_features(df)

Xtr, Xval, ytr, yval = train_test_split(X, y, random_state=1, test_size=0.40)
model = make_pipeline(StandardScaler(), RandomForestRegressor(n_jobs=8, n_estimators=24, verbose=True))

model.fit(Xtr, ytr)
ypred_tr = model.predict(Xtr)
ypred_val = model.predict(Xval)


print(f'Train RMSE: {mean_squared_error(ytr, ypred_tr, squared=False):.30f}')
print(f'Valid RMSE: {mean_squared_error(yval, ypred_val, squared=False):.30f}')
