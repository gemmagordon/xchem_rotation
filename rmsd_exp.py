# get RMSD of each conformer compared to the experimental conformer
import kabsch_functions_new as kabsch
from rdkit import Chem
import pandas as pd
import brute_force_func_coplanar as bf


# load in sdf with conformers 
suppl = Chem.SDMolSupplier('x540_conformers.sdf')
fragment_files, frag_filenames = bf.get_sdfs('Mpro_fragments')
frag_mols = bf.sdf_to_mol(fragment_files)
confs = [] 
confs.append(frag_mols[0]) # add experimental conformer to list of 50 conformers
for mol in suppl:
    confs.append(mol) 
# experimental conformer points
exp_coords = confs[0].GetConformer().GetPositions()

rmsd_vals = []
for mol in confs:
    # get coords
    mol_coords = mol.GetConformer().GetPositions()
    # align to experimental conformer
    
    __, rmsd_val = kabsch.align_coords(mol_coords, exp_coords)
    rmsd_vals.append(rmsd_val)

print(rmsd_vals)

# conformer_results = pd.DataFrame()
# conformer_results['RMSD exp'] = pd.Series(rmsd_vals)
# conformer_results.to_csv('conformer_results_exp.csv')


