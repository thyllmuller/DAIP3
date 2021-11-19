from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import D_C_T
import timeit
import xgboost as xgb
import numpy as np

lbl = preprocessing.LabelEncoder()
tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()

'''XGBoost'''
# Prep the data
cb["State"] = lbl.fit_transform(cb["State"].astype(str))
# cb["State"] = cb["State"].astype("string")
cb_x = cb.drop(["Churn"], axis=1)
cb_y = cb["Churn"]

# Creating split for training and testing
x_train, x_test, y_train, y_test = train_test_split(cb_x, cb_y, test_size=0.3)
data_train = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
data_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
'''
# Determine what model to use via XGBoost:
model = XGBClassifier()
model.fit(x_train, y_train)
print(model.objective)
#binary:logistic
'''

# Run XGBoost
param = {
    'eta': 0.1,
    'max_depth': 3,
    'objective': 'binary:logistic', }

steps = 20  # The number of training iterations

'''Running XGBoost: Notes are at the borrom for params'''

xg_class = xgb.train(param, data_train, steps)
preds = xg_class.predict(data_test)
best_preds = np.asarray([np.argmax(line) for line in preds])
precision = precision_score(y_test, best_preds, average='macro', zero_division=True)
recall = recall_score(y_test, best_preds, average='macro')
accuracy = accuracy_score(y_test, best_preds)
print(str(f"Precision:{precision:.2f}"))
print(str(f"Recall:{recall:.2f}"))
print(str(f"Accuracy:{accuracy:.2f}"))

xg_model = xgb.plot_tree(xg_class, num_trees=2, rankdir='LR')

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

plt.show()
image = xgb.to_graphviz(xg_class)
image.graph_attr = {'dpi': '400'}
image.render('XGBoost', format="png")


# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.

# the leaf value is representative (like raw score) for the probability of the data point belonging to the positive class.
# The final probability prediction is obtained by taking sum of leaf values
# (raw scores) in all the trees and then transforming it between 0 and 1 using a sigmoid function.
# The leaf value (raw score) can be negative, the value 0 actually represents probability being 1/2.
# to calculate leaf probability= 1/(1+np.exp(-1*leaf_value))
print(1 / (1 + np.exp(-1 * 0.136439726)))
