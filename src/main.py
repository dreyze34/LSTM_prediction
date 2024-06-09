import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_processing import *
from prediction import *

def main():
    # Load the data
    dataframe, step = load_data(filename='raw_data.csv')
    
    # Impute missing values
    imputed_dataframe = impute_nan(dataframe)
    
    # Prepare the data for training
    look_back = int(pd.Timedelta(days=1)/step) # 1 day look back
    trainX, trainY, scaler = prepare_data(imputed_dataframe, look_back)
    
    # Build the model
    model = build_model(trainX, trainY)
    
    # Train the model
    model = train_model(model, trainX, trainY)
    
    # Evaluate the model
    trainScore = evaluate_model(model, trainX, trainY, scaler)
    print(f'Training Score (RMSE) : {trainScore}')
    
    # Make predictions
    num_predictions = int(pd.Timedelta(days=1)/step) # 1 day to predict
    futurePredictions = prediction(model, trainX, num_predictions, scaler)

    # Convert the predictions to a DataFrame with datetime index
    predictions_df = convert_prediction(dataframe, futurePredictions)

    # Plot the predictions
    plt.figure(figsize=(15, 5))
    plt.plot(predictions_df)
    plt.xlabel('Time')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title('Energy Consumption Prediction')
    plt.show()

    return predictions_df

if __name__ == '__main__':
    main()