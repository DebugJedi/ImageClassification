from sklearn.datasets import fetch_openml
import pandas as pd
def getData():
    print("getting the mnist data.....")
    mnist = fetch_openml('mnist_784', version= 1)
    print("Here's mnist data.....")
    return mnist

def desc_stats(data):
    marker = "#"*50
    print(marker)
    print("data info....")
    print(data.info(),"\n")
    # print(data.describe())
    print("Total null values...")
    print(data.isnull().sum())
    print(marker,"\n")
    
    

    