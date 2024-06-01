from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class cls():
    def __init__(self, X_train= None,X_test=None, y_train = None,y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    # def scale(self):
    #     self.scaler = StandardScaler()
    #     return self.scaler
    def sgd_clf_ev(self, cv= 3, X=None, y= None):

        print("Initiating SGDClassifier evaluation using cross validation")
        sgd_clf = SGDClassifier(random_state=42)
        y_pred = cross_val_predict(sgd_clf, X=X, y=y, cv=cv)        
        conf_mat = confusion_matrix(y, y_pred)
        print(conf_mat)
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        norm_conf_mat = conf_mat/row_sums
        np.fill_diagonal(norm_conf_mat,0)
        plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
        print("Printing the prediction....")
        plt.show()
        
    
    def rf_clf_ev(self, X=None, y=None, cv =3, random_state=42):
        print("Initiating random forest evaluation using cross validation")
        forest_clf = RandomForestClassifier(random_state=random_state)
        y_pred = cross_val_predict(forest_clf, X=X, y=y, cv=cv)        
        conf_mat = confusion_matrix(y, y_pred)
        row_sums = conf_mat.sum(axis=1, keepdims=True)
        norm_conf_mat = conf_mat/row_sums
        np.fill_diagonal(norm_conf_mat,0)
        plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
        print("Printing the prediction....")
        plt.show()

    def sgd_pred(self, X =None, y=None, cv= 3, random_state = 42):
        sgd_clf = SGDClassifier(random_state=random_state)
        return cross_val_predict(sgd_clf, X, y, cv=cv)

    def precision_recall(self, y_train=None, y_pred=None):
        
        return precision_score(y_train, y_pred), recall_score(y_train, y_pred), f1_score(y_train, y_pred)
     

        
    def grid_search(self,X=None,y=None, params = None, cv=3, score='accuracy', random_state= 42 ):
        rf_clf = RandomForestClassifier(random_state=random_state)
        grid_search = GridSearchCV(rf_clf, param_grid=params, cv=cv, scoring=score)
        grid_search.fit(X,y)
        return grid_search.best_estimator_
