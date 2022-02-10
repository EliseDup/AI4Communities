# Data wrangling
import pandas as pd
import numpy as np

# Deep learning:
#from keras.models import Sequential
#from keras.layers import LSTM, Dense


class LearningModel():
    """
    A class to create a deep time series model
    """

    def __init__(
            self,
            data: pd.DataFrame,
            Y_var: str,
            lag: int,
            LSTM_layer_depth: int,
            epochs=10,
            batch_size=256,
            train_test_split=0
    ):

        self.data = data
        self.Y_var = Y_var
        self.lag = lag
        self.LSTM_layer_depth = LSTM_layer_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_test_split = train_test_split

    @staticmethod
    def create_X_Y(ts: list, lag: int) -> tuple:
        """
        A method to create X and Y matrix from a time series list for the training of
        deep learning models
        """
        X, Y = [], []

        if len(ts) - lag <= 0:
            X.append(ts)
        else:
            for i in range(len(ts) - lag):
                Y.append(ts[i + lag])
                X.append(ts[i:(i + lag)])

        X, Y = np.array(X), np.array(Y)

        # Reshaping the X array to an LSTM input shape
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        return X, Y

    def create_data_for_NN(
            self,
            use_last_n=None
    ):
        """
        A method to create data for the neural network model
        """
        # Extracting the main variable we want to model/forecast
        y = self.data[self.Y_var].tolist()

        # Subseting the time series if needed
        if use_last_n is not None:
            y = y[-use_last_n:]

        # The X matrix will hold the lags of Y
        X, Y = self.create_X_Y(y, self.lag)

        # Creating training and test sets
        X_train = X
        X_test = []

        Y_train = Y
        Y_test = []

        if self.train_test_split > 0:
            index = round(len(X) * self.train_test_split)
            X_train = X[:(len(X) - index)]
            X_test = X[-index:]

            Y_train = Y[:(len(X) - index)]
            Y_test = Y[-index:]

        return X_train, X_test, Y_train, Y_test