import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from math import radians, cos, sin, asin, sqrt
import time
from scipy.spatial import KDTree
from pyproj import Geod
geod = Geod(ellps='WGS84')
#import dataset

### Some of this code is reused from GeoTrackNet

AVG_EARTH_RADIUS = 6378.137  # in km
SPEED_MAX = 30 # knot
FIG_DPI = 150

LAT,LON,SOG,COG,HEAD, ROT,TIMESTAMP,MMSI = range(8) 

############# DATA PROCESSING #################

def filter_data(df, config):
    # Apply filters
    df_filtered = df[
        (df['LAT'] >= config.lat_min) &
        (df['LAT'] <= config.lat_max) &
        (df['LON'] >= config.lon_min) &
        (df['LON'] <= config.lon_max) &
        (df['SOG'] >= config.sog_min) &
        (df['SOG'] <= config.sog_max) &
        (df['COG'] >= config.cog_min) &
        (df['COG'] <= config.cog_max) &
        (df['SHIP_TYPE'].isin(config.ship_types)) &
        (~df['NAV_STAT'].isin(config.nav_stat))
    ]

    # Keep only necessary columns
    df_filtered = df_filtered[['LAT', 'LON', 'SOG', 'COG', 'HEAD', 'ROT', 'TIMESTAMP', 'MMSI']]    

    # Convert timestamp to unix timestamp
    #df_filtered['TIMESTAMP'] = pd.to_datetime(df_filtered['TIMESTAMP'], dayfirst=True).astype(int) / 10**9

    return df_filtered


def interpolate_nans(group):
    """Fill NaNs. Linear for scalar channels; sin/cos interpolation for
    circular channels (COG, HEAD) so 359->1 does not pass through 180."""
    group = group.copy()
    for col in ["LAT", "LON", "SOG", "ROT"]:
        group[col] = group[col].interpolate()
    for col in ["COG", "HEAD"]:
        rad = np.radians(group[col])
        s = np.sin(rad).interpolate()
        c = np.cos(rad).interpolate()
        group[col] = np.degrees(np.arctan2(s, c)) % 360.0
    return group


def split_voyages(traj_data, interval_max=2*3600):
    """
    Split voyages into contiguous ones.
    """
    mmsi_list = []
    voyages = []
    count = 0
    for mmsi in list(traj_data.keys()):
        v = traj_data[mmsi]
        if v.shape[0]>0:
            # Intervals between successive messages in a track
            intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
            idx = np.where(intervals > interval_max)[0]
            if len(idx) == 0:
                voyages.append(v)
                mmsi_list.append(mmsi)
                count += 1
            else:
                tmp = np.split(v,idx+1)
                for t in tmp:
                    voyages.append(t)
                    mmsi_list.append(mmsi)
                    count += 1
    return voyages, mmsi_list

def remove_short_voyages(voyages, mmsi_list, min_length=20, min_duration=4*3600):
    """
    Remove voyages that are too short.
    """
    new_voyages = []
    new_mmsi_list = []
    for i in range(len(voyages)):
        v = voyages[i]
        duration = v[-1,TIMESTAMP] - v[0,TIMESTAMP]
        if (len(v) >= min_length) and (duration >= min_duration):
            new_voyages.append(v)
            new_mmsi_list.append(mmsi_list[i])
    return new_voyages, new_mmsi_list


def find_outliers(coordinates, threshold=0.019):
    # Scale lon by cos(lat) so 1 unit ~ 1 degree of latitude (~111 km)
    # in both axes; otherwise the threshold is ~2x looser east-west
    # than north-south at 56N.
    coords = coordinates.copy()
    coords[:, 1] = coords[:, 1] * np.cos(np.radians(np.mean(coords[:, 0])))
    tree = KDTree(coords)
    distances, _ = tree.query(coords, k=2)

    # The distances array now contains the distance to the nearest and second nearest neighbor for each point
    # We are interested in the second nearest neighbor (i.e., the first neighbor that is not the point itself)
    nearest_neighbor_distances = distances[:, 1]

    # Identify outliers
    outliers = np.where(nearest_neighbor_distances > threshold)[0]

    # Remove outliers
    #cleaned_coordinates = np.delete(coordinates, outliers, axis=0)

    return outliers

def remove_outliers_i(i_voyage):
    i, voyage = i_voyage
    coordinates = voyage[:,:2]
    outlier_indices = find_outliers(coordinates)
    # if len(outlier_indices) > 0:
    #     print(f"Found {len(outlier_indices)} outliers in voyage {i}")
    cleaned_voyage = np.delete(voyage, outlier_indices, axis=0)
    return (i, cleaned_voyage)





############# HELPER FUNCTIONS #################

def _interp_angle_deg(a0, a1, w):
    """Interpolate two angles (degrees) with weight w in [0,1] across the
    0/360 wrap via sin/cos components."""
    a0r, a1r = radians(a0), radians(a1)
    s = (1 - w) * sin(a0r) + w * sin(a1r)
    c = (1 - w) * cos(a0r) + w * cos(a1r)
    return (np.degrees(np.arctan2(s, c))) % 360.0


def interpolate_(t, track):
    """
    Interpolating the AIS message of vessel at a specific "t".
    INPUT:
        - t : 
        - track     : AIS track, whose structure is
                     [LAT, LON, SOG, COG, HEADING, ROT, TIMESTAMP, MMSI]
    OUTPUT:
        - [LAT, LON, SOG, COG, HEADING, ROT, TIMESTAMP, MMSI]
                        
    """
    
    before_p = np.nonzero(t >= track[:,TIMESTAMP])[0]
    after_p = np.nonzero(t < track[:,TIMESTAMP])[0]
   
    if (len(before_p) > 0) and (len(after_p) > 0):
        apos = after_p[0]
        bpos = before_p[-1]    
        # Interpolation
        dt_full = float(track[apos,TIMESTAMP] - track[bpos,TIMESTAMP])
        if (abs(dt_full) > 2*3600):
            return None
        dt_interp = float(t - track[bpos,TIMESTAMP])
        try:
            az, _, dist = geod.inv(track[bpos,LON],
                                   track[bpos,LAT],
                                   track[apos,LON],
                                   track[apos,LAT])
            dist_interp = dist*(dt_interp/dt_full)
            lon_interp, lat_interp, _ = geod.fwd(track[bpos,LON], track[bpos,LAT],
                                               az, dist_interp)
            speed_interp = (track[apos,SOG] - track[bpos,SOG])*(dt_interp/dt_full) + track[bpos,SOG]
            w = dt_interp / dt_full
            course_interp = _interp_angle_deg(track[bpos, COG], track[apos, COG], w)
            heading_interp = _interp_angle_deg(track[bpos, HEAD], track[apos, HEAD], w)  
            rot_interp = (track[apos,ROT] - track[bpos,ROT])*(dt_interp/dt_full) + track[bpos,ROT]
            # if dt_interp > (dt_full/2):
            #     nav_interp = track[apos,NAV_STT]
            # else:
            #     nav_interp = track[bpos,NAV_STT]                             
        except:
            return None
        return np.array([lat_interp, lon_interp,
                         speed_interp, course_interp, 
                         heading_interp, rot_interp, 
                         t,track[0,MMSI]])
    else:
        return None




def trackOutlier(A):
    """
    Koyak algorithm to perform outlier identification
    Our approach to outlier detection is to begin by evaluating the expression
    “observation r is anomalous with respect to observation s ” with respect to
    every pair of measurements in a track. We address anomaly criteria below; 
    assume for now that a criterion has been adopted and that the anomaly 
    relationship is symmetric. More precisely, let a(r,s) = 1 if r and s are
    anomalous and a(r,s) = 0 otherwise; symmetry implies that a(r,s) = a(s,r). 
    If a(r,s) = 1 either one or both of observations are potential outliers, 
    but which of the two should be treated as such cannot be resolved using 
    this information alone.
    Let A denote the matrix of anomaly indicators a(r, s) and let b denote 
    the vector of its row sums. Suppose that observation r is an outlier and 
    that is the only one present in the track. Because we expect it to be 
    anomalous with respect to many if not all of the other observations b(r) 
    should be large, while b(s) = 1 for all s ≠ r . Similarly, if there are 
    multiple outliers the values of b(r) should be large for those observations
    and small for the non-outliers. 
    Source: "Predicting vessel trajectories from AIS data using R", Brian L 
    Young, 2017
    INPUT:
        A       : (nxn) symmatic matrix of anomaly indicators
    OUTPUT:
        o       : n-vector outlier indicators
    
    # FOR TEST
    A = np.zeros((5,5))
    idx = np.array([[0,2],[1,2],[1,3],[0,3],[2,4],[3,4]])
    A[idx[:,0], idx[:,1]] = 1
    A[idx[:,1], idx[:,0]] = 1    sampling_track = np.empty((0, 9))
    for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
        tmp = utils.interpolate(t,v)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    """
    assert (A.transpose() == A).all(), "A must be a symatric matrix"
    assert ((A==0) | (A==1)).all(), "A must be a binary matrix"
    # Initialization
    n = A.shape[0]
    b = np.sum(A, axis = 1)
    o = np.zeros(n)
    while(np.max(b) > 0):
        r = np.argmax(b)
        o[r] = 1
        b[r] = 0
        for j in range(n):
            if (o[j] == 0):
                b[j] -= A[r,j]
    return o.astype(bool)
    
#===============================================================================
#===============================================================================
def detectOutlier(track, speed_max = SPEED_MAX):
    """
    removeOutlier() removes anomalus AIS messages from AIS track.
    An AIS message is considered as beging anomalous if the speed is
    infeasible (> speed_max). There are two types of anomalous messages:
        - The reported speed is infeasible
        - The calculated speed (distance/time) is infeasible
    
    INPUT:
        track       : a (nxd) matrix. Each row is an AIS message. The structure 
                      must follow: [Timestamp, Lat, Lon, Speed]
        speed_max   : knot
    OUTPUT:
        o           : n-vector outlier indicators
    """
    # Remove anomalous reported speed
    o_report = track[:,3] > speed_max # Speed in track is in knot
    if o_report.all():
        return o_report, None
    track = track[np.invert(o_report)]
    # Calculate speed base on (lon, lat) and time
    
    N = track.shape[0]
    # Anomoly indicator matrix
    A = np.zeros(shape = (N,N))
    
    # Anomalous calculated-speed
    for i in range(1,5):
        # the ith diagonal
        _, _, d = geod.inv(track[:N-i,2],track[:N-i,1],
                           track[i:,2],track[i:,1])
        delta_t = track[i:,0] - track[:N-i,0].astype(float)  
        cond = np.logical_and(delta_t > 2,d/delta_t > (speed_max*0.514444))
        abnormal_idx = np.nonzero(cond)[0]
        A[abnormal_idx, abnormal_idx + i] = 1
        A[abnormal_idx + i, abnormal_idx] = 1    

    o_calcul = trackOutlier(A)
            
    return o_report, o_calcul


def resample_voyage(voyage, resolution=600):
    """Resample a voyage to a fixed time grid (default 600 s = 10 min) via
    geodesic interpolation. Required so that sequence step k always means
    'k * 10 minutes' — otherwise per-horizon errors are not comparable
    across trajectories. Returns None if the voyage cannot be resampled."""
    t0, t1 = int(voyage[0, TIMESTAMP]), int(voyage[-1, TIMESTAMP])
    sampled = []
    for t in range(t0, t1 + 1, resolution):
        p = interpolate_(t, voyage)
        if p is None:          # gap > 2 h should not occur post-split; be safe
            return None
        sampled.append(p)
    if len(sampled) < 2:
        return None
    return np.stack(sampled)


def resample_voyage_i(i_voyage, resolution=600):
    i, voyage = i_voyage
    return (i, resample_voyage(voyage, resolution))


