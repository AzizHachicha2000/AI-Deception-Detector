#data_loader.py
import os
import pandas as pd

def load_kaggle_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'deceptive' not in df.columns or 'text' not in df.columns:
            raise ValueError("CSV missing required columns ('text' and/or 'deceptive')")
            
        df['label'] = df['deceptive'].map({'deceptive': 0, 'truthful': 1})
        return df[['text', 'label']].copy()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")