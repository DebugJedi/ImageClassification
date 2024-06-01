from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class process():
    def __init__(self, X= None, y = None):
        self.X = X
        self.y = y
        self.scaler = StandardScaler()

    def splitCLS(self, test_size = 0.3, random_state =42):
        print("initiated split, with test size = {} and random_state = {}".format(test_size, random_state))
        self.X_split_train, self.X_split_test, self.y_split_train, self.y_split_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        
        return self.X_split_train, self.X_split_test, self.y_split_train, self.y_split_test
    
    def scale(self,X= None):
        print("Intiating scaling on train and test set...")
        
        self.X = self.scaler.fit_transform(X)
        
        return self.X