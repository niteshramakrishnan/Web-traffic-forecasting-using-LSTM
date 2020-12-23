from lstm import LstmParam, LstmNetwork
from sklearn import metrics
import numpy as np
import pandas as pd

class SquaredLossLayer:
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

class LSTMModel():
    def __init__ (self, mem_cell, x_dim):
        self.mem_cell_ct = mem_cell
        self.x_dim = x_dim
        self.lstm_param = LstmParam(self.mem_cell_ct, self.x_dim)
        self.lstm_net = LstmNetwork(self.lstm_param)


    def fit (self, X_train, y_train, iterations=100, learning_rate=0.1, noPrint=True):
        self.lstm_net.x_list_clear()
        for cur_iter in range(iterations):
            if not noPrint:
                print("iter", "%2s" % str(cur_iter), end=": ")
            for index in range(len(y_train)):
                self.lstm_net.x_list_add(X_train[index])

            self.y_pred = [self.lstm_net.lstm_node_list[index].lstm_state.hstate[0] for index in range(len(y_train))]

            loss = self.lstm_net.y_list_is(y_train, SquaredLossLayer)
            if not noPrint:
                print("loss:", "%.3e" % loss)
            self.lstm_param.apply_diff(l=learning_rate)
            self.lstm_net.x_list_clear()


    def evaluate(self, true_val, pred_val):
        mse = metrics.mean_squared_error(true_val, pred_val)
        evs = metrics.explained_variance_score(true_val, pred_val)
        r2 = metrics.r2_score(true_val, pred_val)
        msle = metrics.mean_squared_log_error(true_val, pred_val)
        me = metrics.max_error(true_val, pred_val)
        mae = metrics.mean_absolute_error(true_val, pred_val)
        print("Mean Squared Error: ", mse)
        print("Explained Variance Score: ", evs)
        print("Mean Squared Log Error: ", msle)
        print("Max Residual Error: ", me)
        print("R2 Score: ", r2)
        print("Mean Absolute Error: ", mae)
        return {"mse": mse, "evs": evs, "r2": r2, "msle": msle, "me": me, "mae": mae}

    def test(self, X_test, y_test):
        for index in range(len(y_test)):
            self.lstm_net.x_list_add(X_test[index])
        self.y_test_pred = [self.lstm_net.lstm_node_list[index].lstm_state.hstate[0] for index in range(len(y_test))]

    def predictForNDays(self, X, N):
        df = pd.DataFrame(X)
        start = 549
        col = 0
        for i in range(N):
            df[start] = self.predictHelper(df.values)
            start += 1
            df.drop(col, axis=1, inplace=True)
            col += 1
        return df.loc[:, 549:]

    def predictHelper(self, X):
        self.lstm_net.x_list_clear()
        for index in range(len(X)):
            self.lstm_net.x_list_add(X[index])
        return [self.lstm_net.lstm_node_list[index].lstm_state.hstate[0] for index in range(len(X))]

    
