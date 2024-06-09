import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import Huber # type: ignore

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def prepare_data(dataframe, num_steps):

    dataset = dataframe.values
    dataset = dataset.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    trainX, trainY = create_dataset(dataset, num_steps)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    return trainX, trainY, scaler

def build_model(trainX, trainY):
    model = Sequential()
    model.add(Input(shape=(1, trainX.shape[2])))
    model.add(LSTM(100, return_sequences=True, activation='tanh'))
    model.add(LSTM(50, return_sequences=True, activation='tanh'))
    model.add(LSTM(25, activation='tanh'))
    model.add(Dense(1))
    return model

def train_model(model, trainX, trainY):
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=5e-5))
    model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=1)
    return model

def evaluate_model(model, trainX, trainY, scaler):
    trainPredict = model.predict(trainX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    return trainScore

def prediction(model, trainX, num_steps, scaler):
    futurePredictions = []
    current_input = trainX[-1]

    for i in range(num_steps):
        current_input = current_input.reshape((1, 1, num_steps))
        next_pred = model.predict(current_input, verbose=0)
        futurePredictions.append(next_pred[0,0])
        current_input = np.roll(current_input, -1, axis=2)
        current_input[0, 0, -1] = next_pred
        
    futurePredictions = scaler.inverse_transform(np.array(futurePredictions).reshape(-1, 1))    

    return futurePredictions

def convert_prediction(dataframe, futurePredictions):
    futurePredictions = pd.DataFrame(futurePredictions, index=pd.date_range(start=dataframe.index[-1], periods=len(futurePredictions), freq='15T'), columns=['prediction'])
    return futurePredictions