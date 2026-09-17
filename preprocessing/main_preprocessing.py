import os
import sys
from pathlib import Path
from functools import partial
import argparse
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent))

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool
import time
import pickle

from forecasting.config import load_config
from preprocessing.data_utils import interpolate_, split_voyages, remove_short_voyages, filter_data, remove_outliers_i, interpolate_nans, resample_voyage_i

LAT,LON,SOG,COG,HEAD, ROT,TIMESTAMP,MMSI = range(8) 


"""
    Folder Structure is like this:
    raw
    ├── 2019
    │   ├── 2019-01.parquet
    │   ├── 2019-02.parquet
    │   ...
    ├── 2020
    │   ├── 2020-01.parquet
    │   ├── 2020-02.parquet
    │   ...
    preprocessed
    ├── 2019.parquet
    ├── 2020.parquet
    """




def main(config_name, year):
    start_time = time.time()

    
    config = load_config(config_name, "dataset")
    file_paths = glob.glob(config.raw_data_folder + "/" + year + "/*.parquet")
    
    combined_trajs = []
    combined_mmsi_list = []
    for file_path in file_paths:

        df = pd.read_parquet(file_path)
        df = filter_data(df, config)
        df = df.sort_values(["MMSI", "TIMESTAMP"]).reset_index(drop=True)

        # Interpolate to reduce NaN values
        df = df.groupby("MMSI", group_keys=False).apply(interpolate_nans).reset_index(drop=True)

        # Create a dict with MMSI as key and the trajectory as value
        traj_data = {mmsi: group.to_numpy() for mmsi, group in df.groupby('MMSI')} # for 2019: len: 1819

        # Split voyages based on the interval between points
        voyages, mmsi_list = split_voyages(traj_data, interval_max = config.interval_max) # for 2019: len: 6849

        # Remove short voyages
        voyages, mmsi_list = remove_short_voyages(voyages, mmsi_list, min_length=config.min_length, min_duration=config.min_duration) # for 2019: len: 5669

        # Remove long voyages
        # We do this, because long voyages are often just ships that float in the water and barely moving
        # We have this behavior also in smaller than 23h voyages, but less common so we jsut take this as noise in the data.
        #voyages, mmsi_list = zip(*[(voyage, mmsi) for voyage, mmsi in zip(voyages, mmsi_list) if (voyage[-1, TIMESTAMP] - voyage[0, TIMESTAMP]) > config.max_duration])
        # Edit: Last I tested this removed almost all trajectories, so deactivated for now
        
        # Remove outliers
        with Pool() as pool:
            cleaned_voyages = pool.map(remove_outliers_i, enumerate(voyages))

        # Combine the cleaned trajectories with the other features
        cleaned_voyages = sorted(cleaned_voyages, key=lambda x: x[0])  # sort by original index
        cleaned_voyages = [x[1] for x in cleaned_voyages]  # remove the index

        # Resample every voyage to a fixed 10-minute grid (paper: delta_t = 10 min).
        with Pool() as pool:
            resampled = pool.map(partial(resample_voyage_i, resolution=config.resolution), enumerate(cleaned_voyages))
        resampled = sorted(resampled, key=lambda x: x[0])

        kept_voyages, kept_mmsi = [], []
        for (i, v), mmsi in zip(resampled, mmsi_list):
            if v is not None and len(v) >= config.min_length:
                kept_voyages.append(v[:, :7])   # drop trailing MMSI col from interpolate_
                kept_mmsi.append(mmsi)

        combined_trajs.extend(kept_voyages)
        combined_mmsi_list.extend(kept_mmsi)
        
        print(len(voyages))
        print(f"Elapsed time: {time.time() - start_time} seconds")    

    columns = ['LAT', 'LON', 'SOG', 'COG', 'HEAD', 'ROT', 'TIMESTAMP']
    # Create a list of dictionaries, where each dictionary represents a row in the DataFrame
    data = [{col: traj[:, i] for i, col in enumerate(columns)} for traj in combined_trajs]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    df['MMSI'] = combined_mmsi_list

    # Creat COORDS column with List of (LON, LAT) tuples
    df['COORDS'] = df[['LON', 'LAT']].apply(lambda x: list(zip(x['LON'], x['LAT'])), axis=1)

    # Store df as parquet file at config.preprocessed_folder
    df.to_parquet(config.preprocessed_folder + "/" + year + ".parquet")

    print(f"Total elapsed time: {time.time() - start_time} seconds")
    print(f"Number of trajectories: {len(combined_trajs)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-y', '--year', default='2019', help='The year to process') 
    parser.add_argument('-d', '--data', default='dk', help='The data config to use') # dk, 
    args = parser.parse_args()
    main(args.data, args.year)
