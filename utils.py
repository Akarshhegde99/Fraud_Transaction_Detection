import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(df):
    # Assuming 'Time' and 'Amount' need scaling
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    return df
