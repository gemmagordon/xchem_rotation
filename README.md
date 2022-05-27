# overview

## 1. Data preparation
-	Fragalysis Mpro
-	Ground truth (experimental conformer) vs removed coordinates
-	Conformer generation
-	https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654 

## 2.Generating point clouds of pharmacophores
-	Generate larger point cloud of all fragments that bind target
-	Generate second smaller point cloud for query that will be mapped to fragment cloud
-	From SD files of fragments use RDKit to convert to mol objects from which coordinate of pharmacophores can be extracted. 
-	Have only used donors and acceptors. 
-	For fragments concatenate together all points, results in point cloud
-	For query points filter out/exit program if matrix rank smaller than 3 after concatenation of all query pharmacophore points. Filters out coplanar molecules.

## 3. Generating permutations
-	Number of permutations for e.g. this case with 20 fragments producing X pharmacophore points would be XXXX. That’s with only donor and acceptor types.
-	Reducing number of permutations makes sense both for the context of the problem and for computational complexity
-	Two filters to reduce number of permutations: distance and type of pharmacophore
-	There is a maximum distance between pairs of points within a query point cloud. It wouldn’t be logical to include fragment point permutations that exceed this distance, as they won’t be a good solution. Create subpockets within cloud by clustering with a distance threshold of the maximum pairwise distance between query points, with which to generate permutations.
-	Secondly, query points are of certain pharmacophore types. Wouldn’t be logical to include permutations of points of different pharmacophore types.
-	By filtering by distance and pharmacophore type, can significantly reduce number of permutations that need computing. However, the reduction varies significantly (could calculate an example/give example of certain cases/plot max distance vs number of permutations).
-	Can go further and reduce clustered points (with low threshold, 0.5-1) to centroids. Reduces number of points at expense of accuracy of mapping.

## 4. Kabsch alignment
-	Rigid alignment (no scaling)
-	Assumes correspondence between points (hence permutations needed rather than combinations)
-	General cases
-	Calculate centroids
-	Move centroids to origin, get translation vector
-	Compute covariance matrix
-	Compute optimal rotation matrix
-	Apply rotation matrix
-	Add back translation vector
-	Calculate RMSD between aligned points and points mapped to
-	Edge cases, matrix ranks, filter out coplanar cases






### (notes to self)
- set up python env and new conda env when installing rdkit
### 1. download Mpro dataset from fragalysis
- active site structures includes all query molecules and fragments
- active site intersection with 'xchem screen active site' is just fragments that bind mpro - this is what we want to create the point cloud 
- will need to get just query molecules from active site set at some point 
- (when downloading - 'structures in Hit Navigator' 'separate SDFs in subdirectory')
- sdf files have the 3D coordinates for the fragments/molecules
- when we get to aligning the query molecules against fragments we need to remove the coordinates as these show their true alignment (ground truth) *how do you remove coords from mol object but keep structure?*
- but we also need to store the ground truth data for the query molecules so that we can assess performance of alignment algorithms - i.e. will compare coordinates found by algorithm to ground truth (can use RMSD) *compare files of all vs frag only, extract those not in frag only as query molecules* 
- substructure matching, MCS - will need to find which query molecules are very similar to fragments

### 2. getting pharmacophore point cloud from sdf files
- RDKit SDF supplier class? convert sdf file to rdkit mol object
- toy code extract pharmacophore functions
- get point cloud from ph4 coordinates

### 3. generating conformers for query molecules
- rdkit EmbedMultipleConfs, GetConformers()
- some energy minimisation of conformers?
- just use one conformer for now - sdf gives the experimental one and can extract coords with GetPositions()
- need sets of query molecules with and without coords - set with coords as ground truth so we can see alignment performance
- would assume that 

### 4. alignment
- brute force
- ICP, RANSAC 
### 5. evaluate performance
- compare with coords (ground truth) to without coords - can find RMSD between these two?