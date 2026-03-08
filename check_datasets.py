import pandas as pd
import os

datasets = {
    'Fire': 'dataset/final_fire_dataset2.csv',
    'FlashFlood': 'dataset/flash_flood_data.csv',
    'Flood': 'dataset/sampled_flood_data.csv',
    'PVFlood': 'dataset/pluvial_flood_data_balanced.csv',
    'Quake': 'dataset/earthquake_data.csv',
    'Hurricane': 'dataset/hurricane_data.csv',
    'Tornado': 'dataset/tornado_data.csv'
}

for name, path in datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
    else:
        print(f"\n{name}: FILE NOT FOUND - {path}")
