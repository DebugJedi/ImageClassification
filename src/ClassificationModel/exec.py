from Classification.data import getData
from Classification import desc_stats
from Classification import processing
from Classification import classifiers
class execute():
    def __init__(self):
        self.marker = "#"*45
    def get_dataframe(self):
        mnist = getData()
        X,y = mnist['data'], mnist['target']
        return X,y
    
    def descriptive(self, X, y):
        print("Here are the some descriptive stats.....")
        print("1. Dimensions:-")
        print("The dimension for X: {}".format(X.shape))
        print("The dimesion for y: {} \n".format(y.shape) )
        print("Some stats on independent variable(s)...")
        desc_stats(X)
        print("Some stats on dependent variable(s)....")
        desc_stats(y)

    def split(self, X, y):
        split = processing.process(X, y)
        x_train, x_test, y_train, y_test = split.splitCLS(test_size=0.2)
        return x_train, x_test, y_train, y_test
    
    def scaling(self, X=None):
        scale = processing.process()
        X_scaled = scale.scale(X)
        return X_scaled

    def rf_clf(self, X=None, y=None):
        rf = classifiers.cls()
        rf.rf_clf_ev(X,y)

    def sgd_clf(self, X=None, y=None):
        sgd = classifiers.cls()
        sgd.sgd_clf_ev(X,y)

    def sgd_clf_pred(self, X = None , y=None):
        sgd_pred = classifiers.cls()
        return sgd_pred.sgd_pred(X, y)
    
    def pr_recall(self, y_pred=None, y_train=None):
        pr = classifiers.cls()
        return pr.precision_recall(y_train,y_pred)
    
    def rf_grid(self,X=None,y=None, params = None, cv=3, score='accuracy', random_state= 42):
        grid = classifiers.cls()
        return grid.grid_search(X,y, params=params, cv=cv, score=score)
