import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

x_columns = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8',
           'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16',
           'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20']

df_stop1 = pd.read_csv('stop1.csv')
df_rock1 = pd.read_csv('rock1.csv')
df_stop2 = pd.read_csv('stop2.csv')
df_rock2 = pd.read_csv('rock2.csv')

frames = [df_stop1, df_stop2, df_rock1, df_rock2]
df_train = pd.concat(frames)

#这里写错了 还没改
df_train[x_columns] = df_train[x_columns].divide(df_train['x0'], axis=0)
df_train['label'] = df_train['label'].divide(df_train['y0'], axis=0)
df_train['label'] = df_train['label'].astype(int)

train_coordinates = []

for index in df_train.index:
    item_coordinates = []
    for (columnName, columnData) in df_train[x_columns].iloc[index].iteritems():
        item_coordinates.append(columnData)

    train_coordinates.append(item_coordinates)

train_x = np.array(train_coordinates)
train_y = df_train['label'].values

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.25)
print(train_x.shape, train_y.shape) #(1826, 42) (1826,)
print(val_x.shape, val_y.shape) #((203, 42) (203,)
#

# Random Forest
# Preparing of a list of n_estimators values to tune
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 250]
# Prepare a list of depth values
depths = np.linspace(1, 20, 10, endpoint=True)
depths = depths.astype(int)

params_grid = {
    'n_estimators': n_estimators,
    'max_depth': depths
}

# Initialize a random forest classifier
rf_clf = RandomForestClassifier()
clf = GridSearchCV(rf_clf, params_grid, cv=2, verbose=10)

clf.fit(train_x, train_y)

model = clf.best_estimator_
model.fit(train_x, train_y)
print(model.score(val_x, val_y))
