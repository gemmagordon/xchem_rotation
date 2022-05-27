#!/usr/bin/env python
# coding: utf-8

import os
from rdkit import Chem
from geomtry_utils import transform_ph4s
import kabsch_functions_new as kabsch
import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import joblib
from joblib import Parallel, delayed
import brute_force_func_new as bf
import rich
from rich import print as rprint

### SET UP POCKET POINTS
# get pocket fragments
fragment_files, frag_filenames = bf.get_sdfs('Mpro_fragments')
frag_mols = bf.sdf_to_mol(fragment_files)[:5]
rprint('NUM FRAGMENTS:', len(frag_mols))
frag_donor_coords, frag_acceptor_coords = bf.get_coords_fragments(frag_mols)
frag_donor_coords, frag_acceptor_coords, frag_donor_acceptor_coords, \
        (donor_idxs, acceptor_idxs, donor_acceptor_idxs) = bf.clean_ph4_points(frag_donor_coords, frag_acceptor_coords)

print(frag_donor_coords)
## SET UP QUERY POINTS
#query_sdfs, query_filenames = bf.get_sdfs('Mpro_query')
#query_mols = bf.sdf_to_mol(query_sdfs)

# NOTE while testing - use pocket fragments as query mols
query_mols = frag_mols

# get ph4 coords for a single mol
query_donor_coords, query_acceptor_coords = bf.get_coords_query(query_mols[2])
query_donor_coords, query_acceptor_coords, query_donor_acceptor_coords, (q_do_idxs, q_ac_idxs, q_ar_idxs) = bf.clean_ph4_points(query_donor_coords, query_acceptor_coords)

rprint('num query donors', len(query_donor_coords))
rprint('num query acceptors', len(query_acceptor_coords))
rprint('num query don-acc', len(query_donor_acceptor_coords))


# transform points for test
query_donor_coords_trans, query_acceptor_coords_trans, query_donor_acceptor_coords_trans \
        = transform_ph4s([query_donor_coords, query_acceptor_coords, query_donor_acceptor_coords], angleX=np.pi, angleY=0.45, angleZ=0,
                        translateX=1, translateY=23)


transf_concat_list = []
for item in [query_donor_coords_trans, query_acceptor_coords_trans, query_donor_acceptor_coords_trans]:
    if len(item) > 0:
        transf_concat_list.append(item)
query_points_trans = np.concatenate(transf_concat_list)
rprint('NUM trans QUERY POINTS', len(query_points_trans))

query_points_trans = np.concatenate([query_donor_coords_trans, query_acceptor_coords_trans, query_donor_acceptor_coords_trans])

concat_list = []
for item in [query_donor_coords, query_acceptor_coords, query_donor_acceptor_coords]:
    if len(item) > 0:
        concat_list.append(item)
query_points = np.concatenate(concat_list)
rprint('NUM QUERY POINTS', len(query_points))
rprint('QUERY POINTS:', query_points)

qm_aligned, rmsd_val = kabsch.align_coords(query_points_trans, query_points)
print('RESULT:', rmsd_val)
print(qm_aligned)

# # Brute force draft
### CLUSTER POCKET POINTS 
def cluster(data, distance_threshold):

    model = AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=distance_threshold) # 0.5 - 1 
    model.fit_predict(data)
    pred = model.fit_predict(data)
    rprint("Number of clusters found: {}".format(len(set(model.labels_))))
    labels = model.labels_
    #print('Cluster for each point: ', model.labels_)

    return labels

donor_df = bf.create_ph4_df(frag_donor_coords, donor_idxs)
acceptor_df = bf.create_ph4_df(frag_acceptor_coords, acceptor_idxs)
donor_acceptor_df = bf.create_ph4_df(frag_donor_acceptor_coords, donor_acceptor_idxs)


# Find centroids of clusters: 
def get_centroids(ph4_df):

    x_means = ph4_df.groupby(['cluster_label'])['x'].mean()
    y_means = ph4_df.groupby(['cluster_label'])['y'].mean() 
    z_means = ph4_df.groupby(['cluster_label'])['z'].mean()

    cluster_centroids = []
    for x, y, z in zip(x_means, y_means, z_means):
        centroid_coords = [x, y, z]
        cluster_centroids.append(centroid_coords)

    return cluster_centroids


def create_centroid_df(ph4_df):
    '''get arrays of centroid coords for each cluster (by ph4)'''
        
    # gives arrays of centroid coordinates for each cluster
    centroids = np.array(get_centroids(ph4_df))
    centroid_df = pd.DataFrame([centroids[:,0], centroids[:,1], centroids[:,2]]) # no cluster labels
    centroid_df = centroid_df.transpose()
    centroid_df.columns = ['x', 'y', 'z']

    return centroid_df

donor_centroid_df = create_centroid_df(donor_df)
acceptor_centroid_df = create_centroid_df(acceptor_df)
donor_acceptor_centroid_df = create_centroid_df(donor_acceptor_df)

# collect all points together as new pocket points
# create df with centroid points and labels for which ph4 type, so can separate back out different ph4 types later
# NOTE need to make this into function/more flexible

# NOTE for test removing centroids at first to see if works with exact points; should be RMSD=0 
donor_centroid_df = donor_df 
acceptor_centroid_df = acceptor_df
donor_acceptor_centroid_df = donor_acceptor_df

rprint('check donor_df length', len(donor_centroid_df))
rprint('check acc df length', len(acceptor_centroid_df))
rprint('check don-acc df length', len(donor_acceptor_centroid_df))

centroid_dfs = [donor_centroid_df, acceptor_centroid_df, donor_acceptor_centroid_df]

# CREATE DF OF ALL POINTS
pocket_df = bf.create_pocket_df(donor_centroid_df, acceptor_centroid_df, donor_acceptor_centroid_df)
rprint('check pocket_df length', len(pocket_df))

### FILTER BY DISTANCE:
def filter_by_dist(query_points, pocket_df):
    # get max distance for pairwise points of query molecule
    pdist_q = scipy.spatial.distance.pdist(query_points, metric='euclidean')
    max_query_dist = np.max(pdist_q)
    rprint('MAX DIST:', max_query_dist)

    # get possible subpockets of fragment cloud by clustering, use query max dist as threshold
    pocket_points = []
    for x,y,z in zip(pocket_df['x'], pocket_df['y'], pocket_df['z']):
            pocket_point = [x,y,z]
            pocket_points.append(pocket_point)
    pocket_points = np.array(pocket_points)

    cluster_labels = cluster(pocket_points, distance_threshold=max_query_dist) # order not lost ? so use just points in clustering, then append list of labels to existing df with ph4 labels
    pocket_df['cluster_label'] = pd.Series(cluster_labels) # NOTE should check this works ie stays in order/labels not wrong

    return pocket_df


pocket_df = filter_by_dist(query_points, pocket_df)
pocket_df.to_csv('pocket_df.csv')

def generate_permutations(pocket_df):

    '''resort points in clusters back into their ph4 types (label them) then generate permutations'''
    # sort back into ph4 types within main clusters (subpockets)

    ph4_permutations = []

    cluster_groups =  pocket_df.groupby('cluster_label')
    for name, group in cluster_groups:
        # need to separate back into ph4 types
        # group within clusters by ph4 type
        ph4_types = group.groupby('ph4_label')
        # get arrays of point coords from each ph4 type
        donors = []
        acceptors = []
        donor_acceptors = []
        for name, group in ph4_types:
            for x,y,z in zip(group['x'], group['y'], group['z']):
                coords = [x,y,z]
                if name == 'Donor':
                    donors.append(coords)
                elif name == 'Acceptor':
                    acceptors.append(coords)
                elif name == 'Donor-Acceptor':
                    donor_acceptors.append(coords)

    # get possible combinations/permutations within subpocket, restricted by type/numbers of different ph4s in query molecule
    # e.g. first query mol has 4 donors, 1 acceptor, so from frag donor points choose 4, from acceptor points choose 1 (and then get permutations for different correspondences)

        n_query_acceptors = len(query_acceptor_coords)
        n_query_donors = len(query_donor_coords)
        n_query_donor_acceptors = len(query_donor_acceptor_coords)
        args = []
        if n_query_acceptors:
            args.append(itertools.permutations(acceptors, len(query_acceptor_coords)))
        if n_query_donors:
            args.append(itertools.permutations(donors, len(query_donor_coords)))
        if n_query_donor_acceptors:
            args.append(itertools.permutations(donor_acceptors, len(query_donor_acceptor_coords)))
        
        args = tuple(args)

        for permutation in itertools.product(*args):
            permutation = np.concatenate(permutation)
            
            frag_idxs = []
            ph4_idxs = []
            for coords in permutation:
                # find if coords from donor/acceptor/donor-acceptor (removes error if any same for diff fragments?)
                # get index of coords in df after determining ph4 type
                if len(donors) > 0 and np.any(np.all(coords == donors, axis=1)) == True:
                    ph4_idx = 'Donor'
                    # get frag ID from pocket_df by matching to (a) ph4 type (b) coords
                    frag_idx = donor_centroid_df.loc[(donor_centroid_df['x'] == coords[0]) & (donor_centroid_df['y'] == coords[1]) & (donor_centroid_df['z'] == coords[2]), 'ID']
                    frag_idx = list(frag_idx)
                elif len(acceptors) > 0 and np.any(np.all(coords == acceptors, axis=1)) == True:
                    ph4_idx = 'Acceptor'
                    frag_idx = acceptor_centroid_df.loc[(acceptor_centroid_df['x'] == coords[0]) & (acceptor_centroid_df['y'] == coords[1]) & (acceptor_centroid_df['z'] == coords[2]), 'ID']
                    frag_idx = list(frag_idx)
                elif len(donor_acceptors) > 0 and np.any(np.all(coords == donor_acceptors, axis=1)) == True:
                    ph4_idx = 'Donor-Acceptor'
                    frag_idx = donor_acceptor_centroid_df.loc[(donor_acceptor_centroid_df['x'] == coords[0]) & (donor_acceptor_centroid_df['y'] == coords[1]) & (donor_acceptor_centroid_df['z'] == coords[2]), 'ID'] 
                    frag_idx = list(frag_idx)
                ph4_idxs.append(ph4_idx)
                frag_idxs.append(frag_idx)

            permutation = permutation.reshape(-1,3) # change from tuple to array
            points = (permutation, ph4_idxs, frag_idxs)
            ph4_permutations.append(points)

    return ph4_permutations

ph4_permutations = generate_permutations(pocket_df)
rprint('TOTAL PERMUTATIONS:', len(ph4_permutations))

# create df to hold results
results_df = pd.DataFrame()
rmsd_vals = []
qm_aligned_all = []

results = Parallel(n_jobs=2)(delayed(kabsch.align_coords)(query_points=query_points, ref_points=permutation) for permutation, ph4_idxs, frag_idxs in ph4_permutations)

for result in results:
    qm_aligned_all.append(result[0])
    rmsd_vals.append(result[1])

results_df['RMSD'] = pd.Series(rmsd_vals)
results_df['Query'] = pd.Series(qm_aligned_all)

frag_coords = [p[0] for p in ph4_permutations]
results_df['Fragment'] = pd.Series(frag_coords)
# get fragment ID and ph4 type for each point in permutation
ph4_idxs = [p[1] for p in ph4_permutations]
results_df['PH4 types'] = pd.Series(ph4_idxs)
frag_idxs = [p[2] for p in ph4_permutations]
results_df['IDs'] = pd.Series(frag_idxs)

rprint('Best RMSD:', np.min(rmsd_vals))

# Get best result/s:
# get row of df with lowest RMSD to get fragment and query points
best_results = results_df.loc[results_df['RMSD'] == np.min(results_df['RMSD'])]
best_results = pd.DataFrame(best_results, columns=['RMSD', 'Fragment', 'Query', 'PH4 types', 'IDs'])

rprint('ignore donor-acceptor IDs, incorrect')
rprint(best_results)
best_results.to_csv('debug.csv')

for best_result in best_results.itertuples(index=False):

    # plot best result/s (sometimes multiple with same RMSD)
    ax = plt.figure(figsize=(7,7)).add_subplot(projection='3d')
    # plot matrices for comparison 
    ax.scatter3D(best_result[1][:,0], best_result[1][:,1], best_result[1][:,2], label='Fragment points', color='r', s=100, alpha=0.5)
    ax.scatter3D(best_result[2][:,0], best_result[2][:,1], best_result[2][:,2], label='Query aligned', color='b', s=100, alpha=0.5)
    # set plot titles and labels
    ax.legend(loc='upper right')
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    plt.title(str('RMSD =' + str("{:.2f}".format(float(best_result[0])))), fontsize=20)
    plt.show()


