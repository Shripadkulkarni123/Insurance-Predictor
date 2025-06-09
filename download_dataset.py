import pandas as pd
import requests
import os

def download_dataset():
    # URL for the insurance dataset
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    
    try:
        # Download the dataset
        print("Downloading dataset...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the dataset
        with open('data/insurance.csv', 'wb') as f:
            f.write(response.content)
        
        print("Dataset downloaded successfully!")
        
        # Verify the data
        df = pd.read_csv('data/insurance.csv')
        print("\nDataset Preview:")
        print(df.head())
        print("\nDataset Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset() 