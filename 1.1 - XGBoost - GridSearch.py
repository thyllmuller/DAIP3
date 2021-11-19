from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

import D_C_T
import timeit
import xgboost as xgb
import numpy as np

lbl = preprocessing.LabelEncoder()
tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()
xgb_cl = xgb.XGBClassifier()

'''XGBoost'''
# Prep the data
cb["State"] = lbl.fit_transform(cb["State"].astype(str))
# cb["State"] = cb["State"].astype("string")
cb_x = cb.drop(["Churn"], axis=1)
cb_y = cb["Churn"]

xgb.DMatrix(cb_x, label=cb_y)

X_train, X_test, y_train, y_test = train_test_split(cb_x, cb_y, test_size=.1)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# "Learn" the mean from the training data
mean_train = np.mean(y_train)
# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train
# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE is {:.2f}".format(mae_baseline))

# Parameters that we are going to tune.
params = {
    'max_depth': 9,
    'min_child_weight': 7,
    'eta': .3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'binary:hinge',
}

params['eval_metric'] = "mae"

num_boost_round = 999

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
print("Best MAE: {:.2f} with {} rounds".format(
    model.best_score,
    model.best_iteration + 1))

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)

print(cv_results)

gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(9, 12)
    for min_child_weight in range(5, 8)
]
# Define initial best params and MAE
min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
        max_depth,
        min_child_weight))  # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight  # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth, min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

gridsearch_params = [
    (subsample, colsample)
    for subsample in [i / 10. for i in range(7, 11)]
    for colsample in [i / 10. for i in range(7, 11)]
]

min_mae = float("Inf")
best_params = None  # We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
        subsample,
        colsample))  # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample  # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )  # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample, colsample)
        print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

# This can take some time…
min_mae = float("Inf")
best_params = None

for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))
    # We update our parameters
    params['eta'] = eta
    # Run and time CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics=['mae'],
        early_stopping_rounds=10
    )  # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].argmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))
