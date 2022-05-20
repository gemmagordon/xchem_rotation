#!/usr/bin/env python
# coding: utf-8

# need to run for ground truth and without coords (without cheating)

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
fragment_sdfs, frag_filenames = bf.get_sdfs('Mpro_fragments')
frag_mols = bf.sdf_to_mol(fragment_sdfs)
frag_donor_coords, frag_acceptor_coords, frag_aromatic_coords, \
                            (donor_idxs, acceptor_idxs, aromatic_idxs) = bf.get_coords(frag_mols)


### SET UP QUERY POINTS
query_sdfs, query_filenames = bf.get_sdfs('Mpro_query')
query_mols = bf.sdf_to_mol(query_sdfs)

for mol in query_mols:

    # NOTE all assumes only donor and acceptor coords for a mol - make more flexible
    
    # get ph4 coords for a single mol
    query_donor_coords, query_acceptor_coords, query_aromatic_coords = bf.get_coords_query(query_mols[0])
    query_points = np.concatenate([query_donor_coords, query_acceptor_coords]) 

    # cluster pocket points with low threshold
    donor_df = bf.create_ph4_df(frag_donor_coords)
    acceptor_df = bf.create_ph4_df(frag_acceptor_coords)

    # Find centroids of clusters: 
    donor_centroid_df = bf.create_centroid_df(donor_df)
    acceptor_centroid_df = bf.create_centroid_df(acceptor_df)

    # collect all points together as new pocket points
    # create df with centroid points and labels for which ph4 type, so can separate back out different ph4 types later
    centroid_dfs = [donor_centroid_df, acceptor_centroid_df]
    ph4_labels = ['Donor', 'Acceptor']
    pocket_df = bf.create_pocket_df(centroid_dfs, ph4_labels)

    # filter by distance
    max_query_dist, pocket_df = bf.filter_by_dist(query_points, pocket_df)

    # get permutations
    ph4_permutations = bf.generate_permutations(pocket_df)
    print('TOTAL PERMUTATIONS:', len(ph4_permutations))

    # create df to hold results
    results_df = pd.DataFrame()
    rmsd_vals = []
    permutations = [] 
    qm_aligned_all = []

    results = Parallel(n_jobs=2)(delayed(kabsch.align_coords)(permutation, query_points) for permutation in ph4_permutations)
    for result in results:
        rmsd_vals.append(result[1])
        qm_aligned_all.append(result[0])

    results_df['RMSD'] = pd.Series(rmsd_vals)
    results_df['Fragment'] = pd.Series(permutations)
    results_df['Query'] = pd.Series(qm_aligned_all)

    # Get best result/s:
    # get row of df with lowest RMSD to get fragment and query points
    best_results = results_df.loc[results_df['RMSD'] == np.min(results_df['RMSD'])]
    best_results = pd.DataFrame(best_results, columns=['RMSD', 'Fragment', 'Query'])
    # save to csv? could then plot some results easily if needed


# NOTE should create summary results from all mols and reports for each? e.g. which query, which fragment it best mapped to, RMSD
# should save points as sdfs to map back to original fragments/pymol visualise