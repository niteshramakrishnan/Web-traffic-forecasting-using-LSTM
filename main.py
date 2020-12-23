def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from LSTMModel import LSTMModel

def run(mem_cell_ct=100, x_dim=549, iterations=10, lr=0.1, sample=200, num_folds=5):

    data = pd.read_csv('../src/train_1.csv').fillna(0)
    data = data.sample(n=sample, random_state=1)
    page = data['Page']
    data = data.drop('Page',axis = 1)

    X = data[data.columns[:-1]].values
    y = data[data.columns[-1]].values

    model = LSTMModel(mem_cell_ct, x_dim)
    kf = KFold(n_splits=num_folds)
    count = 1
    train_errors = []
    test_errors = []
    for train_index, test_index in kf.split(X):
    #     print("TRAIN:", train_index, "TEST:", test_index)
        print("Split: ", count)
        count += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, iterations, lr)
        print("Train Evaluation: ")
        temp = model.evaluate(y_train, model.y_pred)
        train_errors.append(temp)
        model.test(X_test, y_test)
        print("Test Evaluation: ")
        temp = model.evaluate(y_test, model.y_test_pred)
        test_errors.append(temp)
    print(model.predictForNDays(X_test, 10))

if __name__ == "__main__":
    run()
