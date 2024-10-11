import pandas as pd
import re

def preprocess(text):
    # Basic preprocessing: convert to lowercase, remove special characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def load_data(path):
    # Load the CSV dataset
    return pd.read_csv(path)

def preprocess_data(df):
    # Preprocess both columns of strings
    df['string_1'] = df['string_1'].apply(preprocess)
    df['string_2'] = df['string_2'].apply(preprocess)
    return df
