# explore adaboost ensemble tree depth effect on performance
import pandas as pd
from numpy import mean
from numpy import std
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot, pyplot as plt

import D_C_T
import timeit

lbl = preprocessing.LabelEncoder()
tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()

# get the dataset
cb["Churn"] = cb["Churn"].map({1: True, 0: False})
cb_x = cb.drop(["Churn"], axis=1)   #data
cb_y = cb["Churn"]                  #target
cb_y_lbl = lbl.fit_transform(cb_y)
cb_x["State"] = lbl.fit_transform(cb_x["State"])


# get a list of models to evaluate
def get_models():
    models = dict()
    # explore depths from 1 to 10
    for i in range(1, 50):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return models


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# define dataset
X = cb_x.values
y = cb_y.values
print("We give a negative weight to classifiers with worse worse than 50% !!")

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
    toc = timeit.default_timer()
    #print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
plt.xlabel("Max Depth of Ensemble")
plt.ylabel("Accuracy")

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))


pyplot.show()
