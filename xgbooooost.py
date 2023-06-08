import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv( "soil_weather_data.csv")
X = df.drop(columns=['Value_x'])
y = df.loc[:,['Value_x']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

X_train = X_train.drop(columns=['year','state_name','county_name'])
X_test = X_test.drop(columns=['year','state_name','county_name'])


scalerXST = StandardScaler().fit(X_train)
scaleryST = StandardScaler().fit(y_train)

X_train = scalerXST.transform(X_train)
y_train = scaleryST.transform(y_train)
X_test = scalerXST.transform(X_test)
y_test = scaleryST.transform(y_test)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

params = { 'max_depth': [15, 20,30,50,70],
           'learning_rate': [0.01,0.05,0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [70,80,100,200]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25,
                         verbose=2)


clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))