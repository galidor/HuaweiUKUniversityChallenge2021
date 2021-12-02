import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv


split = 'test'

data_dir = os.path.join('./dataset/task2_data', split)
prefix = 'task2_' + split + '_'

# Load distance data
df_wifi = pd.read_csv(os.path.join(data_dir, prefix + 'estimated_wifi_distances.csv'))

# Load trajectory lookup
with open(os.path.join(data_dir, prefix + 'lookup.json'), 'r') as f:
    traj_lookup = json.load(f)
traj_lookup_int = dict(zip([int(k) for k in traj_lookup.keys()], traj_lookup.values()))

fingerprint_lookup = {}
for k, v in traj_lookup.items():
    if v not in fingerprint_lookup:
        fingerprint_lookup[v] = [int(k)]
    else:
        fingerprint_lookup[v].append(int(k))
keys_list = list(fingerprint_lookup.keys())
id_to_idx = dict(zip(keys_list, list(range(len(keys_list)))))

# Add trajectory number to dataframe
t1s_wifi = np.array([
    traj_lookup_int[id1] for id1 in df_wifi.id1
], dtype=np.int)
t2s_wifi = np.array([
    traj_lookup_int[id2] for id2 in df_wifi.id2
], dtype=np.int)

df_wifi['t1'] = t1s_wifi
df_wifi['t2'] = t2s_wifi
tup_list = list(zip(list(df_wifi.t1.values), list(df_wifi.t2.values)))
tup_sorted = [tuple(sorted(t)) for t in tup_list]
df_wifi['traj_tup'] = tup_sorted

# Read elevations
elevations = pd.read_csv(os.path.join(data_dir, prefix + 'elevations.csv'))

t1s = np.array([
    traj_lookup_int[id1] for id1 in elevations.id1
], dtype=np.int)
t2s = np.array([
    traj_lookup_int[id2] for id2 in elevations.id2
], dtype=np.int)

# Add trajectory info to elevations
elevations['t1'] = t1s
elevations['t2'] = t2s


# Start clustering by dividing fingerprints according to elevations into separate clusters
fingerprint_clusters = []
for row_idx, row in tqdm(elevations.iterrows()):
    end_cluster_fprint = row.id1
    start_cluster_fprint = row.id2
    if row_idx == 0:
        fingerprint_clusters.append([])
        fingerprint_clusters[-1] += list(range(end_cluster_fprint + 1))
        fingerprint_clusters.append([start_cluster_fprint])
    elif row_idx == (len(elevations) - 1):
        fingerprint_clusters[-1] += list(
            range(
                fingerprint_clusters[-1][-1],
                end_cluster_fprint + 1
            )
        )
        fingerprint_clusters.append(
            list(
                range(
                    start_cluster_fprint,
                    len(traj_lookup)
                )
            )
        )
    else:
        fingerprint_clusters[-1] += list(
            range(
                fingerprint_clusters[-1][-1],
                end_cluster_fprint + 1
            )
        )
        fingerprint_clusters.append([start_cluster_fprint])

# Go from fingerprints to trajectories
traj_clusters = []
for c in fingerprint_clusters:
    c_trajectory = [
        traj_lookup_int[fingerprint] for fingerprint in c
    ]
    c_trajectory = sorted(list(set(c_trajectory)))
    traj_clusters.append(c_trajectory)

traj_to_clust = dict()
for i, c in enumerate(traj_clusters):
    for t in c:
        traj_to_clust[t] = i

# Add initial clusters info to dataframe
c1s = [traj_to_clust[t] for t in df_wifi.t1]
c2s = [traj_to_clust[t] for t in df_wifi.t2]
df_wifi['c1'] = c1s
df_wifi['c2'] = c2s

elevation_traj_tuples = list(zip(list(elevations.t1), list(elevations.t2)))

threshold = 1.0

# Get all distances below the threshold
traj_below_t = df_wifi[(df_wifi.estimated_distance < threshold)]
clusters_to_merge = []

# Merge clusters if they share points that are close enough
for row_idx, row in tqdm(traj_below_t.iterrows()):
    c1_c_idx = None
    c2_c_idx = None
    for i, mc in enumerate(clusters_to_merge):
        if row.c1 in mc:
            c1_c_idx = i
        if row.c2 in mc:
            c2_c_idx = i
    if c1_c_idx is None and c2_c_idx is None:
        if row.c1 == row.c2:
            clusters_to_merge.append([row.c1])        
        else:
            clusters_to_merge.append([row.c1, row.c2])        
    elif c1_c_idx is not None and c2_c_idx is None:
        clusters_to_merge[c1_c_idx].append(row.c2)
    elif c1_c_idx is None and c2_c_idx is not None:
        clusters_to_merge[c2_c_idx].append(row.c1)
    else:
        if c1_c_idx != c2_c_idx:
            clusters_to_merge[c1_c_idx] += clusters_to_merge[c2_c_idx]
            del clusters_to_merge[c2_c_idx]

new_clusters = []
for c_new in clusters_to_merge:
    new_clusters.append([])
    for idx in c_new:
        new_clusters[-1] += traj_clusters[idx]

new_clusters_sorted = []
for c in new_clusters:
    new_clusters_sorted.append(sorted(c))
    
with open("submission.csv", "w", newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(new_clusters_sorted)