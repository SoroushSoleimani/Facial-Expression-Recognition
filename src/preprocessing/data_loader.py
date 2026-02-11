import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(csv_path, image_dir, batch_size=64, target_size=(48, 48)):
    df = pd.read_csv(csv_path)
    df['filename'] = df['filename'].astype(str)
    df['label'] = df['label'].astype(str)
    
    return None, None, None