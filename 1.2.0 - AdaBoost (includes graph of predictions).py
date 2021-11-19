import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import D_C_T
import timeit

lbl = preprocessing.LabelEncoder()
tic = timeit.default_timer()
pd.set_option('display.max_columns', None)
cb = D_C_T.c_Bload()
rng = np.random.RandomState(1)

'''ADABOOST''' "Boosting-based Ensemble learning: sequential learning technique"

# Prep the data
cb["Churn"] = cb["Churn"].map({1: True, 0: False})
cb_x = cb.drop(["Churn"], axis=1)  # data
cb_y = cb["Churn"]  # target
cb_y_lbl = lbl.fit_transform(cb_y)
cb_x["State"] = lbl.fit_transform(cb_x["State"])

# Creating split for training and testing
x_train, x_test, y_train, y_test = train_test_split(cb_x, cb_y, test_size=0.3)  # 70% training and 30% test

dtc = DecisionTreeClassifier(max_depth=3)
ada_model = AdaBoostClassifier(base_estimator=dtc, n_estimators=100)
ada_model = ada_model.fit(x_train.values, y_train.values)
ytest_pred = ada_model.predict(x_test.values)

# Predicting test data and checking the accuracy
ytest_pred = ada_model.predict(x_test.values)
print(confusion_matrix(y_test, ytest_pred))

precision = precision_score(y_test.values, ytest_pred, average='macro', zero_division=True)
recall = recall_score(y_test.values, ytest_pred, average='macro', zero_division=True)
accuracy = accuracy_score(y_test.values, ytest_pred)

print(str(f"Precision:{precision:.2f}"))
print(str(f"Recall:{recall:.2f}"))
print(str(f"Accuracy:{accuracy:.2f}"))

# plotting the result
fig = plt.figure(1, figsize=(30, 8))
ax = fig.add_subplot()
ax.set_ylabel("Churn")
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ytest_pred, lw=0.8, color="red", label="predicted")
plt.legend()

toc = timeit.default_timer()
print(str(f"Time taken to complete task: {(toc - tic):.2f} seconds."))

plt.show()
fig.tight_layout()
fig.savefig('AdaBoost.png')
