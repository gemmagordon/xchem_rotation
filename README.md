
# 1. download Mpro dataset from fragalysis
- active site structures includes all query molecules and fragments
- active site intersection with 'xchem screen active site' is just fragments that bind mpro - this is what we want to create the point cloud 
- will need to get just query molecules from active site set at some point 
- (when downloading - 'structures in Hit Navigator' 'separate SDFs in subdirectory')
- sdf files have the 3D coordinates for the fragments/molecules
- when we get to aligning the query molecules against fragments we need to remove the coordinates as these show their true alignment (ground truth)
- but we also need to store the ground truth data for the query molecules so that we can assess performance of alignment algorithms - i.e. will compare coordinates found by algorithm to ground truth (can use RMSD)

# 2. getting pharmacophore point cloud from sdf files
- RDKit SDF supplier class? convert sdf file to rdkit mol object
- toy code extract pharmacophore functions


# 3. generating conformers for query molecules
- rdkit EmbedMultipleConfs, GetConformers()
- some energy minimisation of conformers?

# 4. alignment
# 5. evaluate performance