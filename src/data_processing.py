import numpy as np
import pandas as pd
import os

def load_data(filename):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', filename))
    dataframe = pd.read_csv(file_path, usecols=[0, 1], index_col=0)
    dataframe.index = pd.to_datetime(dataframe.index)

    first_valid_index = dataframe.first_valid_index()
    dataframe = dataframe.loc[first_valid_index:]
    step = dataframe.index[1] - dataframe.index[0]

    return dataframe, step

def impute_nan(dataframe):

    imputed_dataframe = dataframe.copy()
    imputed_dataframe['weekday'] = imputed_dataframe.index.weekday

    for household in imputed_dataframe.columns:
        if household == 'weekday':
            continue
        column = imputed_dataframe[household]
        median_values = column.groupby([column.index.weekday, column.index.time]).transform('median')
        std_dev = median_values.std()
        noise = np.random.normal(scale=std_dev, size=len(column))
        imputed = median_values + noise
        imputed = imputed.apply(lambda x: x**1)
        min_original_value = column.min()
        imputed = imputed.apply(lambda x: max(min_original_value, x))
        imputed_values = column.fillna(imputed)
        imputed_dataframe[column.name] = imputed_values
    
    imputed_dataframe.drop('weekday', axis=1, inplace=True)

    imputed_dataframe.to_csv('data/processed/processed_data.csv')

    return imputed_dataframe