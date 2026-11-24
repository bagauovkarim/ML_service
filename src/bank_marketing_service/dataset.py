import numpy as np
import pandas as pd
from bank_marketing_service.preprocessing import prepare_features

def load_training_data(path):
    
    df = pd.read_csv(path, sep=";")

    df = df.drop_duplicates()
    
    df["y"] = np.where(df["y"] == "yes", 1, 0)
    
    df = df.drop(columns=["duration"])

    y = df["y"]

    X = df.drop(columns=["y"])

    X = prepare_features(X)
    
    return X, y