import os
from queue import Empty
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

# FUNCTIONS NEEDED TO RUN BRUTE FORCE ALIGNMENT

# ### Retrieve SDFs from directory (downloaded from Fragalysis)
def get_sdfs(dir_name):
    
    sdf_files = []
    filenames = []

    # collect sdf files from dirs
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".sdf"):
                # ignore combined sdf file
                if 'Mpro_combined' not in file:
                    sdf_files.append(os.path.join(root, file))
                    filenames.append(file)

    return sdf_files, filenames


# ### Convert SDFs to rdkit mol objects
# get mol object for each sdf file
def sdf_to_mol(sdf_file_list):
   
   mols = []
   for sdf_file in sdf_file_list:
    suppl = Chem.SDMolSupplier(sdf_file)
    for mol in suppl:
        if mol is None: continue
        mols.append(mol)

   return mols


# ### Extract pharmacophores and their types and coordinates
# code to generate pharmacophores
# feature factory defines set of pharmacophore features being used 
_FEATURES_FACTORY, _FEATURES_NAMES = [], []

def get_features_factory(features_names, resetPharmacophoreFactory=False):

    global _FEATURES_FACTORY, _FEATURES_NAMES

    if resetPharmacophoreFactory or (len(_FEATURES_FACTORY) > 0 and _FEATURES_FACTORY[-1] != features_names):
        _FEATURES_FACTORY.pop()
       # _FEATURES_FACTORY.pop() # NOTE repeated line?
    if len(_FEATURES_FACTORY) == 0:
        feature_factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        _FEATURES_NAMES = features_names
        if features_names is None:
            features_names = list(feature_factory.GetFeatureFamilies())

        _FEATURES_FACTORY.extend([feature_factory, features_names])

    return _FEATURES_FACTORY


def getPharmacophoreCoords(mol, features_names=["Acceptor", "Donor", "Aromatic"], confId=-1):

    # find features of a molecule
    feature_factory, keep_featnames = get_features_factory(features_names)
    rawFeats = feature_factory.GetFeaturesForMol(mol, confId=confId)
    featsDict = defaultdict(list)
    idxsDict = defaultdict(list)

    for f in rawFeats:
        if f.GetFamily() in keep_featnames:
            featsDict[f.GetFamily()].append(np.array(f.GetPos(confId=f.GetActiveConformer())))
            idxsDict[f.GetFamily()].append(np.array(f.GetAtomIds()))

    new_feats_dict = {}
    for key in featsDict:
        new_feats_dict[key] = np.concatenate(featsDict[key]).reshape((-1,3))
    
    return new_feats_dict, idxsDict


# compute pharmacophores coordinates
# NOTE should change just to work for one molecule rather than looping through
def get_coords(mols):

    donor_coords, acceptor_coords, aromatic_coords, donor_acceptor_coords = [], [], [], []
    donor_idxs, acceptor_idxs, aromatic_idxs, donor_acceptor_idxs = [], [], [], []

    for i, mol in enumerate(mols):

        # .get() will just fill with None if key doesn't exist
        pharma_coords, __ = getPharmacophoreCoords(mol)
        _donor_coord = pharma_coords.get('Donor')
        _acceptor_coord = pharma_coords.get('Acceptor')
        aromatic_coord = pharma_coords.get('Aromatic')

        if _donor_coord is not None and _acceptor_coord is not None:
            # get common coords from donors and acceptors and add to donor-acceptor group
            donor_acceptor_coord = [x for x in _donor_coord if x in _acceptor_coord]
            # reduce donor and acceptor coords to only unique values (ie not donor-acceptors)
            donor_coord = [x for x in _donor_coord if x not in _acceptor_coord]
            acceptor_coord = [x for x in _acceptor_coord if x not in _donor_coord]

            donor_acceptor_coords.append(np.array(donor_acceptor_coord))
            donor_coords.append(np.array(donor_coord))
            acceptor_coords.append(np.array(acceptor_coord))

    # remove None values
    _donor_coords = []
    for i, x in enumerate(donor_coords):
        if x is not None:
            _donor_coords.append( x )
            donor_idxs += [i] * len(x)
    donor_coords = _donor_coords

    _acceptor_coords = []
    for i, x in enumerate(acceptor_coords):
        if x is not None:
            _acceptor_coords.append( x )
            acceptor_idxs += [i] * len(x)
    acceptor_coords = _acceptor_coords

    _donor_acceptor_coords = []
    for i, x in enumerate(donor_acceptor_coords):
        if x is not None:
            _donor_acceptor_coords.append( x )
            donor_acceptor_idxs += [i] * len(x)
    donor_acceptor_coords = _donor_acceptor_coords

    _aromatic_coords = []
    for i, x in enumerate(aromatic_coords):
        if x is not None:
            _aromatic_coords.append( x )
            aromatic_idxs += [i] * len(x)
    aromatic_coords = _aromatic_coords

    if len(donor_coords) == 0:
        donor_coords = np.array(donor_coords).reshape(-1,3)
    else:
        donor_coords = [x for x in donor_coords if len(x)!= 0]
        donor_coords = np.concatenate(donor_coords)
        donor_idxs = np.array(donor_idxs)

    if len(acceptor_coords) == 0:
        acceptor_coords = np.array(acceptor_coords).reshape(-1,3)
    else:
        acceptor_coords = [x for x in acceptor_coords if len(x)!= 0]
        acceptor_coords = np.concatenate(acceptor_coords)
        acceptor_idxs = np.array(acceptor_idxs)

    if len(donor_acceptor_coords) == 0:
        donor_acceptor_coords = np.array(donor_acceptor_coords).reshape(-1,3)
    else:
        donor_acceptor_coords = [x for x in donor_acceptor_coords if len(x)!= 0]
        donor_acceptor_coords = np.concatenate(donor_acceptor_coords)
        donor_acceptor_idxs = np.array(donor_acceptor_idxs)

    if len(aromatic_coords) == 0:
        aromatic_coords = np.array(aromatic_coords).reshape(-1,3)
    else:
        aromatic_coords = [x for x in aromatic_coords if len(x)!= 0]
        aromatic_coords = np.concatenate(aromatic_coords)
        aromatic_idxs = np.array(aromatic_idxs)

    return donor_coords, acceptor_coords, aromatic_coords, donor_acceptor_coords, (donor_idxs, acceptor_idxs, aromatic_idxs, donor_acceptor_idxs)


def get_coords_query(mol):

    # .get() will just fill with None if key doesn't exist
    pharma_coords, __ = getPharmacophoreCoords(mol)
    _donor_coords = pharma_coords.get('Donor')
    _acceptor_coords = pharma_coords.get('Acceptor')
    aromatic_coords = pharma_coords.get('Aromatic')

    # get common coords from donors and acceptors and add to donor-acceptor group
    if _donor_coords is not None and _acceptor_coords is not None:
        donor_acceptor_coords = [i for i in _donor_coords if i in _acceptor_coords]
        if len(donor_acceptor_coords) > 0:
            donor_acceptor_coords = np.concatenate(donor_acceptor_coords)
        # reduce donor and acceptor coords to only unique values (ie not donor-acceptors)
        donor_coords = [i for i in _donor_coords if i not in _acceptor_coords]
        acceptor_coords = [i for i in _acceptor_coords if i not in _donor_coords]

        # remove None values
        if donor_coords is not None:
            donor_coords = [x for x in donor_coords if x is not None]
            donor_coords = np.concatenate([donor_coords])
        else:
            donor_coords = np.array([]).reshape(-1,3)
        
        if acceptor_coords is not None:
            acceptor_coords = [x for x in acceptor_coords if x is not None]
            acceptor_coords = np.concatenate([acceptor_coords])
        else:
            acceptor_coords = np.array([]).reshape(-1, 3)
        
        if donor_acceptor_coords is not None:
            donor_acceptor_coords = [x for x in donor_acceptor_coords if x is not None]
            donor_acceptor_coords = np.concatenate([donor_acceptor_coords]).reshape(-1,3)
        else:
            donor_acceptor_coords = np.array([]).reshape(-1, 3)


    if aromatic_coords is not None:
        aromatic_coords = [x for x in aromatic_coords if x is not None]
        aromatic_coords = np.concatenate([aromatic_coords])
    else:
        aromatic_coords = np.array([]).reshape(-1, 3)


    return donor_coords, acceptor_coords, aromatic_coords, donor_acceptor_coords


# ### Setting up fragment point cloud

def plot_coords(donor_coords, acceptor_coords, aromatic_coords, donor_acceptor_coords):
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')

    # visualize the 3D cloud of fragment pharmacophores. They are a good representation of the protein pocket.
    labels = ['Donor', 'Acceptor', 'Aromatic', 'Donor-Acceptor']
    for coords, label in zip([donor_coords, acceptor_coords, aromatic_coords, donor_acceptor_coords], labels):
        if len(coords) != 0:
            ax.scatter3D(coords[:,0], coords[:,1], coords[:,2], label=label)

    plt.legend()
    plt.show()

    return

def cluster(data, distance_threshold):

    model = AgglomerativeClustering(linkage='average', n_clusters=None, distance_threshold=distance_threshold) # 0.5 - 1 
    model.fit_predict(data)
    pred = model.fit_predict(data)
    print("Number of clusters found: {}".format(len(set(model.labels_))))
    labels = model.labels_
    print('Cluster for each point: ', model.labels_)

    return labels


def create_ph4_df(ph4_coords):
    '''cluster ph4 coords and add to df with labels for which cluster each point is in'''
    # cluster donor coords and add with labels to df
    labels = cluster(ph4_coords, distance_threshold=1)
    ph4_df = pd.DataFrame([ph4_coords[:,0], ph4_coords[:,1], ph4_coords[:,2], labels])
    ph4_df = ph4_df.transpose()
    ph4_df.columns = ['x', 'y', 'z', 'cluster_label']

    return ph4_df


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


def create_pocket_df(centroid_dfs):

    labelled_dfs =[]

    for centroid_df in centroid_dfs:
        if centroid_df is donor_centroid_df:
            centroid_df['ph4_label'] = 'Donor'
        elif centroid_df is acceptor_centroid_df:
            centroid_df['ph4_label'] = 'Acceptor'
        elif centroid_df is donor_acceptor_centroid_df:
            centroid_df['ph4_label'] = 'Donor-Acceptor'

        labelled_dfs.append(centroid_df)

    pocket_df = pd.concat(labelled_dfs, axis=0)
    
    return pocket_df


def filter_by_dist(query_points, pocket_df):
# get max distance for pairwise points of query molecule
    pdist_q = scipy.spatial.distance.pdist(query_points, metric='euclidean')
    max_query_dist = np.max(pdist_q)
    print(max_query_dist)

    # get possible subpockets of fragment cloud by clustering, use query max dist as threshold
    pocket_points = []
    for x,y,z in zip(pocket_df['x'], pocket_df['y'], pocket_df['z']):
            # NOTE bug here, separation in donor/acceptor not adding up properly
            pocket_point = [x,y,z]
            pocket_points.append(pocket_point)
    pocket_points = np.array(pocket_points)

    cluster_labels = cluster(pocket_points, distance_threshold=max_query_dist) # order not lost ?? so use just points in clustering, then append list of labels to existing df with ph4 labels
    pocket_df['cluster_label'] = pd.Series(cluster_labels) # NOTE should check this works ie stays in order/labels not wrong
    print(pocket_df)

    return max_query_dist, pocket_df


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
            permutation = np.concatenate(permutation) # change from tuple to array
            ph4_permutations.append(permutation)

    return ph4_permutations

