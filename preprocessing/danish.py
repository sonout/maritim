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
from preprocessing.data_utils import interpolate_, split_voyages, remove_short_voyages, filter_data, remove_outliers_i



LAT,LON,SOG,COG,HEAD, ROT,TIMESTAMP,MMSI = range(8) 


def load_data(file_path):
    """
    Load and preprocess data from the given file path.
    """
    df = pd.read_csv(file_path)

    # Rename columns
    df = df.rename(columns={"Latitude": "LAT", "Longitude": "LON", "# Timestamp": "TIMESTAMP", "Navigational status": "NAV_STAT", "Heading": "HEAD", "Ship type": "SHIP_TYPE"})
    return df
    
def load_and_filter(file_path, config):
    df = load_data(file_path)
    return filter_data(df, config)




def main(year):
    start_time = time.time()

    """
    Folder Structure is like this
    raw
    ├── 2019
    │   ├── 2019-01.csv
    │   ├── 2019-02.csv
    │   ├── 2019-03.csv
    │   ...
    ├── 2020
    │   ├── 2020-01.csv
    │   ├── 2020-02.csv
    │   ├── 2020-03.csv
    │   ...
    preprocessed
    ├── 2019.parquet
    ├── 2020.parquet
    """
    config = load_config("dk", "dataset")
    file_paths = glob.glob(config.raw_data_folder + "/" + year + "/*.csv")

    # Create a partial function for multi threading
    load_and_filter_partial = partial(load_and_filter, config=config)

    # Process files in batches
    combined_trajs = []
    combined_mmsi_list = []
    for i in range(0, len(file_paths), config.batch_size):
        batch_file_paths = file_paths[i:i+config.batch_size]
        
        # This part will be done in parallel
        with Pool() as p:
            res_dfs = p.map(load_and_filter_partial, batch_file_paths)

        # Concat all loaded DFs and sort
        df = pd.concat(res_dfs)
        df = df.sort_values(by=['MMSI', 'TIMESTAMP'])

        # Interpolate to reduce NaN values
        df = df.groupby('MMSI').apply(lambda group: group.interpolate()).reset_index(drop=True)

        # Create a dict with MMSI as key and the trajectory as value
        traj_data = {mmsi: group.to_numpy() for mmsi, group in df.groupby('MMSI')}

        # Split voyages based on the interval between points
        voyages, mmsi_list = split_voyages(traj_data, interval_max = config.interval_max)

        # Remove short voyages
        voyages, mmsi_list = remove_short_voyages(voyages, mmsi_list, min_length=config.min_length, min_duration=config.min_duration)

        # Remove long voyages
        # We do this, because long voyages are often just ships that float in the water and barely moving
        # We have this behavior also in smaller than 23h voyages, but less common so we jsut take this as noise in the data.
        #voyages = [idx for idx in range(len(voyages)) if (voyages[idx][-1, TIMESTAMP] - voyages[idx][0, TIMESTAMP]) > config.max_duration]
        voyages, mmsi_list = zip(*[(voyage, mmsi) for voyage, mmsi in zip(voyages, mmsi_list) if (voyage[-1, TIMESTAMP] - voyage[0, TIMESTAMP]) > config.max_duration])

        print(len(voyages))
        print(f"Elapsed time: {time.time() - start_time} seconds")

        # Remove outliers
        with Pool() as pool:
            cleaned_voyages = pool.map(remove_outliers_i, enumerate(voyages))

        # Combine the cleaned trajectories with the other features
        cleaned_voyages = sorted(cleaned_voyages, key=lambda x: x[0])  # sort by original index
        cleaned_voyages = [x[1] for x in cleaned_voyages]  # remove the index

        # TODO: Add maxium length and duration and split accordingly
        # A: For now we remove time greater than 23h as those are mostly Floaters 
        # A: We will check how the length works with ML Models, and than maybe shorten length as well or whatever.
        # TODO: Normalize?
        # A: We can do that alter if we see necessary


        combined_trajs.extend(cleaned_voyages)
        combined_mmsi_list.extend(mmsi_list)

    

    columns = ['LAT', 'LON', 'SOG', 'COG', 'HEAD', 'ROT', 'TIMESTAMP']
    # Create a list of dictionaries, where each dictionary represents a row in the DataFrame
    data = [{col: traj[:, i] for i, col in enumerate(columns)} for traj in combined_trajs]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    df['MMSI'] = combined_mmsi_list

    # Store df as parquet file at config.preprocessed_folder
    df.to_parquet(config.preprocessed_folder + "/" + year + ".parquet")


    print(f"Total elapsed time: {time.time() - start_time} seconds")
    print(f"Number of trajectories: {len(combined_trajs)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-y', '--year', required=True, help='The year to process')
    args = parser.parse_args()
    main(args.year)



# def sample_voyages_DEPRECATED(voyages, resolution=5*60):
#     """
#     Sample voyages at the given resolution.
#     """
#     Vs = dict()
#     count = 0
#     for k, v in tqdm(list(voyages.items())):
#         sampling_track = np.empty((0, 8))
#         for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), resolution):
#             tmp = interpolate_(t,v)
#             if tmp is not None:
#                 sampling_track = np.vstack([sampling_track, tmp])
#             else:
#                 sampling_track = None
#                 break
#         if sampling_track is not None:
#             Vs[count] = sampling_track
#             count += 1
#     return Vs

# def dict_to_list(voyages):
#     """
#     Convert a dictionary of voyages to a list.
#     """
#     return [v for k, v in voyages.items()]


# def splitting_voyages_DEPRECATED(voyages, max_duration = 24):
#     # Note: Because of downsampling to 5min freq: 1h = 12 * 5mins 
#     print('Re-Splitting...')
#     Data = dict()
#     count = 0
#     for k in tqdm(list(voyages.keys())): 
#         v = voyages[k]
#         # Split AIS track into small tracks whose duration <= 1 day
#         idx = np.arange(0, len(v), 12*max_duration)[1:]
#         tmp = np.split(v,idx)
#         for subtrack in tmp:
#             # only use tracks whose duration >= 4 hours
#             if len(subtrack) >= 12*4:
#                 Data[count] = subtrack
#                 count += 1
#     return Data


# def normalization_DEPRECATED(Data):
#     ## STEP 9: NORMALISATION
#     #======================================
#     print('Normalisation...')
#     for k in tqdm(list(Data.keys())):
#         v = Data[k]
#         v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
#         v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
#         v[:,SOG][v[:,SOG] > SOG_MAX] = SOG_MAX
#         v[:,SOG] = v[:,SOG]/SOG_MAX
#         v[:,COG] = v[:,COG]/360.0
#     return Data
