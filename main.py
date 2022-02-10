import DataReader
from LearningModel import LearningModel
from sklearn.neural_network import MLPClassifier



train_test_split = 0.10
lag = 24
LSTM_layer_depth = 64
epochs = 10

name = "data/consumption/donneeconso01.csv"
data = DataReader.data_analysis(name, "1D")
model = LearningModel(
    data=data,
    Y_var='conso',
    lag=lag,
    LSTM_layer_depth=LSTM_layer_depth,
    epochs=epochs,
    train_test_split=train_test_split) # The share of data that will be used for validation

# Getting the data
X_train, X_test, Y_train, Y_test = model.create_data_for_NN()


nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples, nx*ny))
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(d2_train_dataset, Y_train)
nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples, nx*ny))

res = clf.predict(d2_test_dataset)
print(res)