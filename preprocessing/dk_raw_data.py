import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import glob


from multiprocessing import Pool


sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))
from forecasting.config import load_config


"""
    Folder Structure is like this
    raw
    ├── 2019
    │   ├── 2019-01-01.csv
    │   ├── 2019-01-02.csv
    │   ├── ...
    │   ├── 2019-02-01.csv
    │   ...
    ├── 2020
    │   ├── 2020-01-01.csv
    │   ├── 2020-01-02.csv
    │   ├── ...
    │   ├── 2020-02-01.csv
    │   ...
"""

"""
    After this script we have
    raw
    ├── 2019
    │   ├── 2019-01.parquet
    │   ├── 2019-02.parquet
    │   ...
    ├── 2020
    │   ├── 2020-01.parquet
    │   ├── 2020-02.parquet
    │   ...
"""


def load_data(file_path):
    """
    Load and preprocess data from the given file path.
    """
    df = pd.read_csv(file_path)

    # Rename columns
    df.rename(columns={"Latitude": "LAT", "Longitude": "LON", "# Timestamp": "TIMESTAMP", "Navigational status": "NAV_STAT", "Heading": "HEAD", "Ship type": "SHIP_TYPE", "Length": "LENGTH", "Width": "WIDTH"}, inplace=True)
    return df


def filter_perprocess(df):
    
    df = df.reindex(columns=['TIMESTAMP', 'MMSI', 'LAT', 'LON', 'NAV_STAT', 'ROT', 'SOG', 'COG','HEAD', 'IMO', 'SHIP_TYPE', 'WIDTH', 'LENGTH'])
    #df.drop(columns=['Cargo type','Type of mobile','Name', 'Type of position fixing device', 'Callsign', 'Draught','Destination','ETA','Data source type','A', 'B', 'C', 'D'], inplace=True)

    df['SHIP_TYPE'] = df['SHIP_TYPE'].replace('Undefined', np.nan)
    df['NAV_STAT'] = df['NAV_STAT'].replace('Unknown value', np.nan)

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True).astype(int) / 10**9

    df['ROT'] = df['ROT'].astype('float32')
    df['SOG'] = df['SOG'].astype('float32')
    df['COG'] = df['COG'].astype('float32')
    df['HEAD'] = df['HEAD'].astype('float32')
    df['LENGTH'] = df['LENGTH'].fillna(0).astype('int32')
    df['WIDTH'] = df['WIDTH'].fillna(0).astype('int32')
    df['TIMESTAMP'] = df['TIMESTAMP'].astype('int32')

    return df
    

def load_and_filter(file_path):
    df = load_data(file_path)
    return filter_perprocess(df)


def main(year, start_month, end_month):
    
    config = load_config("dk", "dataset")

    # Get filenames for each month
    filename_batches = [
        glob.glob(f"{config.raw_data_folder}/{year}/aisdk-{year}-{month:02d}-*.csv") +
        glob.glob(f"{config.raw_data_folder}/{year}/aisdk_{year}{month:02d}*.csv")
        for month in range(start_month, end_month+1)
    ]

    for month, filenames in enumerate(filename_batches, start=start_month):
        start_time = time.time()
        print(f"Processing month {month}")
        with Pool() as p:
            res_dfs = p.map(load_and_filter, filenames)

        # Concat all loaded DFs and sort
        df = pd.concat(res_dfs)
        del res_dfs
        df = df.sort_values(by=['MMSI', 'TIMESTAMP'])
        df.to_parquet(f"{config.raw_data_folder}/{year}/{year}-{month:02d}.parquet")
        del df
        print(f"Elapsed time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-y', '--year', default="2015", help='The year to process')
    parser.add_argument('-s', '--start_month', type=int, default=1, help='Start month (1-12)')
    parser.add_argument('-e', '--end_month', type=int, default=12, help='End month (1-12)')
    args = parser.parse_args()
    main(args.year, args.start_month, args.end_month)
