from rdkit import Chem
# from rdkit.Chem import MolFromSmiles
import networkx as nx
from itertools import islice
import numpy as np
import rdkit
from rdkit.Chem import rdmolfiles, rdmolops, BRICS, Recap
# import openbabel as ob
from rdkit.Chem import Draw

def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atoms = []
    features = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            atoms.append(atom.GetSymbol().lower())
        else:
            atoms.append(atom.GetSymbol())
        feature = atom_features(atom)
        features.append(feature / sum(feature))


    return atoms

def smile_to_graph_recap(smile,count_re):
    mol = Chem.MolFromSmiles(smile)

    submols = mol.GetSubstructMatches(Chem.MolFromSmarts('[!R][R]'))

    c_size, features, edge_index, atoms = smile_to_graph(smile)
    if len(submols) == 0 :
        return c_size, features, edge_index, atoms, count_re

    subbonds = [mol.GetBondBetweenAtoms(x, y) for x, y in submols]
    id = 0
    atom_id = 0

    while (c_size - atom_id) / c_size > 0.85 and id < len(submols):
        bond_id = subbonds[id].GetIdx()
        atom_id = max(subbonds[id].GetEndAtomIdx(), subbonds[id].GetBeginAtomIdx())

        if (c_size - atom_id) / c_size < 0.25:
            return c_size, features, edge_index, atoms, count_re
        if 0.5 <= (c_size - atom_id) / c_size <= 0.75:
            break
        id += 1

    for i in range(atom_id):
        features[i] = np.zeros(len(features[i]))

    edges = []
    bonds = mol.GetBonds()
    for bond in bonds:
        if bond.GetIdx() > bond_id:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # 显示结果
    # bonds_id = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in submols]
    # if len(bonds_id) > 0:
    #     frags = Chem.FragmentOnBonds(mol, bonds_id[:1])
    #     type(frags)
    #     smis = Chem.MolToSmiles(frags)
    #     smis = smis.split('.')
    #     mols = []
    #     for smi in smis:
    #         mols.append(smi)
    #     smile_re = max(mols, key=len, default='')
    # c_size_re, features_re, edge_index_re, atoms_re = smile_to_graph(smile_re)
    return c_size - atom_id, features, edge_index, atoms[atom_id:], count_re+1


# hierarch = Recap.RecapDecompose(mol)
# # mol_leave = list(hierarch.GetLeaves().keys())
# mol_children = list(hierarch.GetAllChildren().keys())
# smile_re = max(mol_children, key=len, default='')
# if len(mol_children) == 0:
#     smile_re = smile

# mol_re = Chem.MolFromSmiles(smile_re)
# for i in mol.GetAtoms():
#     i.SetIntProp("atom_idx", i.GetIdx())
# for i in mol.GetBonds():
#     i.SetIntProp("bond_idx", i.GetIdx())
#  all_bonds_idx = [bond.GetIdx() for bond in mol.GetBonds()]

def smiles2adjoin(smiles,explicit_hydrogens=True,canonical_atom_order=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        # mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u,v] = 1.0
        adjoin_matrix[v,u] = 1.0
    return atoms_list,adjoin_matrix