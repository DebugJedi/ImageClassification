"""Main module."""
from exec import execute
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import cross_val_predict



prc = execute()

X, y = prc.get_dataframe()

prc.descriptive(X,y)

X_train, X_test, y_train, y_test = prc.split(X,y)
print("Dimensions of X_test set: {}".format(X_test.shape))
print("Dimension of X_train set: {}".format(X_train.shape))

X_train_scaled = prc.scaling(X_train.astype(np.float64))
X_test_scaled = prc.scaling(X_test.astype(np.float64))

print("Dimensions of X_train_scaled: {}".format(X_train_scaled.shape))
print("Dimensions of X_test_scaled: {}".format(X_test_scaled.shape))

y_pred = prc.sgd_clf_pred(X_train_scaled, y_train)

precision, recall , f1Score = prc.pr_recall(y_pred, y_train)



# y_pred = cross_val_predict(SGDClassifier(random_state=42), X_train_scaled, y_train)



# prc.sgd_clf(X_train_scaled, y_train)

# params = {
#     'max_depth': [3,20, None],
#     'n_estimators': [10, 40],
#     'max_features': [11,70],
#     'min_samples_split': [3]
# }

# print(prc.rf_grid(X=X_train, y=y_train, params=params))

# print("Select your model: \n")
# print("rf --> random Forest \n")
# print("sgd --> SGD Classifier \n")
# val = input("Enter your selection here: ")
# if val == "rf":
#     prc.rf_clf(X_train_scaled, y_train)
# elif val == "sgd":
#     prc.sgd_clf(X_train_scaled, y_train)
# else:
#     print("Unidentified selection. Try again...")


