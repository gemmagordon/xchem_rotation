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
import brute_force_func_coplanar as bf
import rich
from rich import print as rprint
from rdkit.Chem import Draw


### SET UP POCKET POINTS
# get pocket fragments
fragment_files, frag_filenames = bf.get_sdfs('Mpro_fragments')
frag_mols = bf.sdf_to_mol(fragment_files)
#frag_mols = frag_mols[1:2]

frag_donor_coords, frag_acceptor_coords = bf.get_coords_fragments(frag_mols)
frag_donor_coords, frag_acceptor_coords, (donor_idxs, acceptor_idxs) = bf.clean_ph4_points(frag_donor_coords, frag_acceptor_coords)

# NOTE while testing - use pocket fragments as query mols 
# # NOTE CONFORMER TESTS query_mols = conformers of one fragment (x540)
#query_mols = frag_mols 
query_mols = []
query_mols.append(frag_mols[0]) # add experimental conformer to list of query conformers

suppl = Chem.SDMolSupplier('x540_50_confs.sdf')
for x in suppl: # using 50 conformers
    query_mols.append(x)

# FOR EACH CONFORMER, RUN ALGORITHM (WITHOUT CREATING CENTROIDS, so using all original fragment points)
all_best_rmsd = []
for mol in query_mols:

    # get ph4 coords for a single mol
    results = bf.get_coords_query(mol)
    if results is None:
        raise ValueError('Not valid query. ph4s are coplanar')
    else:
        query_donor_coords, query_acceptor_coords = results

    query_donor_coords, query_acceptor_coords, (q_do_idxs, q_ac_idxs) = bf.clean_ph4_points(query_donor_coords, query_acceptor_coords)


    concat_list = []
    for item in [query_donor_coords, query_acceptor_coords]:
        if len(item) > 0:
            concat_list.append(item)
    query_points = np.concatenate(concat_list)


    # # Brute force draft
    ### CLUSTER POCKET POINTS 
    def cluster(data, distance_threshold):

        model = AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=distance_threshold) # 0.5 - 1 
        model.fit_predict(data)
        pred = model.fit_predict(data)
        #rprint("Number of clusters found: {}".format(len(set(model.labels_))))
        labels = model.labels_
        #print('Cluster for each point: ', model.labels_)

        return labels

    donor_df = bf.create_ph4_df(frag_donor_coords, donor_idxs)
    acceptor_df = bf.create_ph4_df(frag_acceptor_coords, acceptor_idxs)
    #rprint(acceptor_df.columns)

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
        centroid_df['ID'] = ph4_df['ID']

        return centroid_df

    donor_centroid_df = create_centroid_df(donor_df)
    acceptor_centroid_df = create_centroid_df(acceptor_df)


    # collect all points together as new pocket points
    # create df with centroid points and labels for which ph4 type, so can separate back out different ph4 types later
    # NOTE need to make this into function/more flexible

    # NOTE for test removing centroids at first to see if works with exact points; should be RMSD=0 
    # REMOVED NO CENTROIDS
    donor_centroid_df = donor_df 
    acceptor_centroid_df = acceptor_df

    # CREATE DF OF ALL POINTS
    pocket_df = bf.create_pocket_df(donor_centroid_df, acceptor_centroid_df)

    ### FILTER BY DISTANCE:
    def filter_by_dist(query_points, pocket_df):
        # get max distance for pairwise points of query molecule
        # print(query_points.shape)
        assert query_points.shape[0] > 2, 'Error, too few points'
        pdist_q = scipy.spatial.distance.pdist(query_points, metric='euclidean')
        # print(pdist_q)
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
            if n_query_donors:
                args.append(itertools.permutations(donors, len(query_donor_coords)))
            if n_query_acceptors:
                args.append(itertools.permutations(acceptors, len(query_acceptor_coords)))

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
    all_best_rmsd.append(np.min(rmsd_vals))


rprint('ignore donor-acceptor IDs, incorrect')

conformer_results = pd.DataFrame()
conformer_results['RMSD ph4'] = pd.Series(all_best_rmsd)
conformer_results.to_csv('conformer_results.csv')
print(all_best_rmsd)

