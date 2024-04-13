import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filename):
    return pd.read_csv(filename)

def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['Result'] = label_encoder.fit_transform(data['Result'])
    X = data.drop('Result', axis=1)
    y = data['Result']
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)
