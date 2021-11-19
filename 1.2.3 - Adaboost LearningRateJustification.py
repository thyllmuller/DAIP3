# explore adaboost ensemble number of trees effect on performance
import pandas as pd
from numpy import mean
from numpy import std
from numpy import arange
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot, pyplot as plt

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


# get a list of models to evaluate
def get_models():
    models = dict()
    # explore learning rates from 0.1 to 2 in 0.1 increments
    for i in (0.001, 0.01, 0.1, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8):
        key = '%.3f' % i
        models[key] = AdaBoostClassifier(learning_rate=i)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X.values, y.values, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X = cb_x
y = cb_y
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(model, X, y)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=45)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

pyplot.show()
