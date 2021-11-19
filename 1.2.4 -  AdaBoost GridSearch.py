# example of grid searching key hyperparameters for adaboost on a classification dataset
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

import D_C_T
import timeit

lbl = preprocessing.LabelEncoder()
tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()

# get the dataset
cb["Churn"] = cb["Churn"].map({1: True, 0: False})
cb_x = cb.drop(["Churn"], axis=1)  # data
cb_y = cb["Churn"]  # target
cb_y_lbl = lbl.fit_transform(cb_y)
cb_x["State"] = lbl.fit_transform(cb_x["State"])

# define dataset
X = cb_x
y = cb_y
model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [5, 7, 10, 50, 100, 500]
grid['learning_rate'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X.values, y.values)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))
