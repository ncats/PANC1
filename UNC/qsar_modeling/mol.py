from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import DataStructs
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize as sklearn_normalize

#largely for memoization of descriptors without having to lug them around the
#program
class Mol:


    def __init__(self, mol, score, identifier = None):

        self.mol = mol
        self.score = score
        self.descriptor_dict = {}
        self.identifier = identifier
        self.inchi = None

    @classmethod
    def from_smiles(cls, smiles, score, identifier = None):
        mol = Chem.MolFromSmiles(smiles)
        return cls(mol = mol, score = score, identifier = identifier)

    #skip concatenation for MUDRA
    def get_descriptor(self, descriptor_function, concatenate = True):

        if descriptor_function in self.descriptor_dict:
            descriptor = self.descriptor_dict[descriptor_function]

        else:
            descriptor = descriptor_function(self)
            #descriptor = descriptor_function(self, rdkit_mol = False)
            self.descriptor_dict[descriptor_function] = descriptor

        if concatenate and isinstance(descriptor, tuple):
            descriptor = np.concatenate(descriptor)

        return descriptor

    def get_inchi(self):
        if not self.inchi:
            self.inchi = Chem.MolToInchi(self.mol)

        return self.inchi

    def get_similarity(self, other_mol, descriptor_function):
        d1 = self.get_descriptor(descriptor_function)
        d2 = other_mol.get_descriptor(descriptor_function)

        return tanimoto_similarity(d1, d2)

    def __str__(self):

        return f"{Chem.MolToSmiles(self.mol)}: {self.score}"

    def __repr__(self):

        return self.__str__()

    def __eq__(self, other):
        
        a = self.get_inchi()
        b = other.get_inchi()

        return (a == b)

    def __ne__(self, other):

        return not self.__eq__(other)

    def __hash__(self):

        return self.get_inchi().__hash__()

def get_descriptors(mols, descriptor_function, normalize = False):

        descriptors = []
        scores = []
        for mol in mols:
            #descriptors.append(mol.get_descriptor(descriptor_function, concatenate = concatenate))
            print(descriptor_function)
            descriptors.append(mol.get_descriptor(descriptor_function))
            scores.append(mol.score)

        descriptors = np.array(descriptors)
        scores = np.array(scores)

        if(normalize):
            descriptors = sklearn_normalize(descriptors, axis = 0, norm = 'l2')

        return (descriptors, scores)

#return lists of Mol objects
def read_sdf(sdf_file, score_column_name, pos_class_name = None, neg_class_name
        = None, relation_name = None, take_negative_log = False, mol_column_name = 'ROMol', shuffle = True):

    df = PandasTools.LoadSDF(sdf_file)

    df_precise = pd.DataFrame() #empty dataframes for checking if None later
    df_imprecise = pd.DataFrame()

    if relation_name:
        relations = df[relation_name]
        exact_indices = (relations == "=")
        df_precise = df[exact_indices]
        df_imprecise = df[~exact_indices]

    else:
        df_precise = df

    if not pos_class_name:
        precise_scores = df_precise[score_column_name].tolist()
        precise_scores = [float(x) for x in precise_scores]
        precise_scores = np.array(precise_scores)

        if(not df_imprecise.empty):
            imprecise_scores = df_imprecise[score_column_name].tolist()
            imprecise_scores = np.array(imprecise_scores, dtype = float)
            relations = df_imprecise[relation_name]
            if(take_negative_log):
                imprecise_scores = -np.log10(imprecise_scores / 10**9)
                inverted_relations = []
                for relation in relations:
                    if relation == "<":
                        inverted_relations.append(">")
                    elif relation == ">":
                        inverted_relations.append("<")
                relations = inverted_relations

            imprecise_scores = np.array([r + str(v) for r,v in zip(relations, imprecise_scores)])

    #make score list by parsing score_column_name
    else:
        score_names = df[score_column_name].tolist()
        scores = []
        i = 0
        for score in score_names:
            if score == pos_class_name:
                scores.append(1)
            elif score == neg_class_name:
                scores.append(0)
            else:
                raise Exception(f"Score value {score} at index {i} does not match provided names ({pos_class_name} and {neg_class_name})")
            i = i + 1

        precise_scores = np.array(scores)

    #check for zeros in precise scores
    if not pos_class_name:
        zero_scores = (precise_scores == 0)
        if (sum(zero_scores) > 0):
            raise Exception(f"Some score values are zero: {np.argwhere(zero_scores == True)}")

    precise_mols = np.array(df_precise[mol_column_name].tolist())
    imprecise_mols = None
    if(not df_imprecise.empty):
        imprecise_mols = np.array(df_imprecise[mol_column_name].tolist())

    if(shuffle):
        np.random.seed(42)
        p = np.random.permutation(len(precise_mols))
        precise_mols = precise_mols[p]
        precise_scores = precise_scores[p]

        if imprecise_mols is not None:
            p = np.random.permutation(len(imprecise_mols))
            imprecise_mols = imprecise_mols[p]
            imprecise_scores = imprecise_scores[p]

    if take_negative_log: #assume nanomolar
        precise_scores = -np.log10(precise_scores / 10**9)

    assert len(precise_mols) == len(precise_scores), f"Different number of molecules and scores: {len(precise_mols)} vs. {len(precise_scores)}"
    if(imprecise_mols is not None):
        assert len(imprecise_mols) == len(imprecise_scores), f"Different number of molecules and scores: {len(imprecise_mols)} vs. {len(imprecise_scores)}"

    precise_return_list = []
    for mol, score in zip(precise_mols, precise_scores):
        m = Mol(mol, score)
        precise_return_list.append(m)

    if imprecise_mols is not None:
        imprecise_return_list = []
        for mol, score in zip(imprecise_mols, imprecise_scores):
            m = Mol(mol, score)
            imprecise_return_list.append(m)
    else:
        imprecise_return_list = None

    #return (precise_mols, precise_scores, imprecise_mols, imprecise_scores)
    return precise_return_list, imprecise_return_list

def read_sdf_efficiently(filename, score_column_name, skip_first_mol = False,
        limit = -1):

    mols = []

    sdm = SDMolSupplier(filename)

    for i, mol in enumerate(sdm):

        if skip_first_mol and i == 0:
            continue

        if limit > 0 and i >= limit:
            break

        m = Mol(mol, float(mol.GetProp(score_column_name)))
        mols.append(m)
        print(f"{i} mols read\r", end = "")

    return(mols)

def write_sdf(mols, filename):

    f = Chem.SDWriter(filename)

    for mol in mols:
        m = mol.mol
        m.SetProp("Activity", str(mol.score))
        f.write(m)


def tanimoto_similarity(d1, d2):

    both = np.sum(np.bitwise_and(d1,d2))
    only_one = np.sum(np.bitwise_xor(d1,d2)) 
    return both / (only_one + both)



