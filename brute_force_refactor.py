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
import brute_force_func_new as bf
import scipy
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import joblib
from joblib import Parallel, delayed


### SET UP POCKET POINTS
# get pocket fragments
fragment_files, frag_filenames = bf.get_sdfs('Mpro_fragments')
frag_mols = bf.sdf_to_mol(fragment_files)[:5]
frag_donor_coords, frag_acceptor_coords, frag_aromatic_coords, \
                            (donor_idxs, acceptor_idxs, aromatic_idxs) = bf.get_coords(frag_mols)


### SET UP QUERY POINTS
#query_sdfs, query_filenames = bf.get_sdfs('Mpro_query')
#query_mols = bf.sdf_to_mol(query_sdfs)

# NOTE while testing - use pocket fragments as query mols
query_mols = frag_mols

# get ph4 coords for a single mol
query_donor_coords, query_acceptor_coords, query_aromatic_coords = bf.get_coords_query(query_mols[0])

# transform points for test
query_donor_coords_trans, query_acceptor_coords_trans, query_aromatic_coords_trans \
    = transform_ph4s([query_donor_coords, query_acceptor_coords, query_aromatic_coords], angleX=np.pi, angleY=0.45, angleZ=0,
                     translateX=1, translateY=23)

query_points = np.concatenate([query_donor_coords_trans, query_acceptor_coords_trans])
#query_points = np.concatenate([query_donor_coords, query_acceptor_coords])

#qm_aligned, rmsd_val = kabsch.align_coords(query_points_trans, query_points)
#print('RESULT:', rmsd_val)
#print(qm_aligned)

# # Brute force draft
### CLUSTER POCKET POINTS 

def cluster(data, distance_threshold):

    model = AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=distance_threshold) # 0.5 - 1 
    model.fit_predict(data)
    pred = model.fit_predict(data)
    print("Number of clusters found: {}".format(len(set(model.labels_))))
    labels = model.labels_
    #print('Cluster for each point: ', model.labels_)

    return labels

def create_ph4_df(ph4_coords):
    '''cluster ph4 coords and add to df with labels for which cluster each point is in'''
    # cluster donor coords and add with labels to df
    labels = cluster(ph4_coords, distance_threshold=1)
    ph4_df = pd.DataFrame([ph4_coords[:,0], ph4_coords[:,1], ph4_coords[:,2], labels])
    ph4_df = ph4_df.transpose()
    ph4_df.columns = ['x', 'y', 'z', 'cluster_label']

    return ph4_df

donor_df = create_ph4_df(frag_donor_coords)
acceptor_df = create_ph4_df(frag_acceptor_coords)


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

# collect all points together as new pocket points
# create df with centroid points and labels for which ph4 type, so can separate back out different ph4 types later
# NOTE need to make this into function/more flexible

# NOTE for test removing centroids at first to see if works with exact points; should be RMSD=0 
donor_centroid_df = donor_df 
acceptor_centroid_df = acceptor_df
print('check donor_df length', len(donor_centroid_df))
print('check acc df length', len(acceptor_centroid_df))

centroid_dfs = [donor_centroid_df, acceptor_centroid_df]


def create_pocket_df(centroid_dfs):

    labelled_dfs =[]

    for centroid_df in centroid_dfs:
        if centroid_df is donor_centroid_df:
            centroid_df['ph4_label'] = 'Donor'
        elif centroid_df is acceptor_centroid_df:
            centroid_df['ph4_label'] = 'Acceptor'

        labelled_dfs.append(centroid_df)

    pocket_df = pd.concat(labelled_dfs, axis=0)
    
    return pocket_df

pocket_df = create_pocket_df(centroid_dfs)
print('check pocket_df length', len(pocket_df))
print(pocket_df.columns)


### FILTER BY DISTANCE:
def filter_by_dist(query_points, pocket_df):
    # get max distance for pairwise points of query molecule
    pdist_q = scipy.spatial.distance.pdist(query_points, metric='euclidean')
    max_query_dist = np.max(pdist_q)
    print('MAX DIST:', max_query_dist)

    # get possible subpockets of fragment cloud by clustering, use query max dist as threshold
    pocket_points = []
    for x,y,z in zip(pocket_df['x'], pocket_df['y'], pocket_df['z']):
            pocket_point = [x,y,z]
            pocket_points.append(pocket_point)
    pocket_points = np.array(pocket_points)

    cluster_labels = cluster(pocket_points, distance_threshold=max_query_dist) # order not lost ? so use just points in clustering, then append list of labels to existing df with ph4 labels
    pocket_df['cluster_label'] = pd.Series(cluster_labels) # NOTE should check this works ie stays in order/labels not wrong

    return max_query_dist, pocket_df


max_query_dist, pocket_df = filter_by_dist(query_points, pocket_df)
print(len(pocket_df))

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
        for name, group in ph4_types:
            print(name, len(group))  # checked - gives totals of acceptors/donors expected
            for x,y,z in zip(group['x'], group['y'], group['z']):
                coords = [x,y,z]
                if name == 'Donor':
                    donors.append(coords)
                elif name == 'Acceptor':
                    acceptors.append(coords)

    # get possible combinations/permutations within subpocket, restricted by type/numbers of different ph4s in query molecule
    # e.g. first query mol has 4 donors, 1 acceptor, so from frag donor points choose 4, from acceptor points choose 1 (and then get permutations for different correspondences)

        n_query_acceptors = len(query_acceptor_coords)
        n_query_donors = len(query_donor_coords)
        args = []
        if n_query_acceptors:
            args.append(itertools.permutations(acceptors, len(query_acceptor_coords)))
        if n_query_donors:
            args.append(itertools.permutations(donors, len(query_donor_coords)))
        args = tuple(args)

        for permutation in itertools.product(*args):
            permutation = np.concatenate(permutation) # change from tuple to array
            ph4_permutations.append(permutation)

    return ph4_permutations

ph4_permutations = generate_permutations(pocket_df)
print('TOTAL PERMUTATIONS:', len(ph4_permutations))


# NOTE but want to do this instead, exactly the same but running parallel

# create df to hold results
results_df = pd.DataFrame()
rmsd_vals = []
qm_aligned_all = []

results = Parallel(n_jobs=2)(delayed(kabsch.align_coords)(query_points=query_points, ref_points=permutation) for permutation in ph4_permutations)
print(results[0])

for result in results:
    qm_aligned_all.append(result[0])
    rmsd_vals.append(result[1])

results_df['RMSD'] = pd.Series(rmsd_vals)
results_df['Fragment'] = pd.Series(ph4_permutations)
results_df['Query'] = pd.Series(qm_aligned_all)
print('best RMSD:', np.min(rmsd_vals))

# Get best result/s:
# get row of df with lowest RMSD to get fragment and query points
best_results = results_df.loc[results_df['RMSD'] == np.min(results_df['RMSD'])]
best_results = pd.DataFrame(best_results, columns=['RMSD', 'Fragment', 'Query'])
print(len(best_results))

for best_result in best_results.itertuples(index=False):

    print(best_result[2])
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


