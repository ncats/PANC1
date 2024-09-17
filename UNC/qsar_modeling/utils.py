import torch
import numpy as np
import pandas as pd
from mol import Mol
from mol import get_descriptors
from mol import write_sdf
from sklearn import metrics
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDMolSupplier
from itertools import product
from rdkit.Chem import Descriptors, Crippen, MolSurf, Lipinski, Fragments, EState, GraphDescriptors
import os
import time

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

#for storing already-calculated descriptors in memory
#could extend to some kind of read-on-demand for huge descriptor sets
class DescriptorHolder:

    #assumes order of descriptors in file is same as 'mols'
    #will have to overhaul in case of column 1 being the molecule id
    def __init__(self, mols, filename, delimiter = ","):

        f = open(filename)
        descs = []
        for i, line in enumerate(f):
            if i == 0:
                continue
            s = line.split(delimiter)
            desc = np.array(s, dtype = int)
            descs.append(desc)

        matrix = np.array(descs)
        self.matrix = matrix
        print(self.matrix.shape)
        ids = [mol.identifier for mol in mols]
        self.descriptor_dictionary = {}
        for i, identifier in enumerate(ids):
            self.descriptor_dictionary[identifier] = i

    def get_descriptor(self, mol):

        desc = self.matrix[self.descriptor_dictionary[mol.identifier]]
        return desc

    #convoluted way to get descriptor for single mixture
    #consequence of the way it's handled in 'models.py'
    #should revisit
    def get_mixture_descriptor(self, mol_a, mol_b):

        identifier = mol_a.identifier + "|" + mol_b.identifier
        pos = self.descriptor_dictionary[mol_a.identifier + "|" + mol_b.identifier]
        desc = self.matrix[pos]
        return desc


def plot_hist(values, title, filename, nbins = 100, xlim = None):

    plt.figure(figsize = (8,6))
    plt.hist(values, bins = nbins)
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.savefig(filename, dpi = 300)
    plt.close()

#values equal to or above cutoff become 1, values below cutoff become 0
def discretize(x, cutoff = 0.5):

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype = float)

    return torch.ge(x, cutoff).squeeze()

def pick_classification_pivot(precise_mols, imprecise_mols = None, pivot = None, num_stds = 0):

    precise_scores = np.array([m.score for m in precise_mols])
    imprecise_scores = np.array([m.score for m in imprecise_mols])

    if not pivot:
        med = np.median(precise_scores)
        std = np.std(precise_scores)
        pivot = med + (num_stds * std)

    new_scores = np.where(precise_scores >= pivot, 1, 0)

    #make new mol list to return
    return_mols = []
    for mol, score in zip(precise_mols, new_scores):
        m = Mol(mol.mol, score)
        return_mols.append(m)

    if imprecise_mols:
        kept_mols = []
        kept_scores = []
        for mol, score in zip(imprecise_mols, imprecise_scores):
            relation = score[0]
            value = float(score[1:])
            if (relation == ">" and value >= pivot) or (relation == "<" and value <= pivot):
                m = Mol(mol.mol, 0)
                return_mols.append(m)

    print(f"Pivot: {pivot}")
    return return_mols

def classification_statistics(actual, predicted, to_string = False):

    cm = metrics.confusion_matrix(actual, predicted)

    accuracy = metrics.accuracy_score(actual, predicted).item()
    f1 = metrics.f1_score(actual, predicted).item()
    Kappa = metrics.cohen_kappa_score(actual, predicted, weights='linear').item()

    TN, FP, FN, TP = cm.ravel()
    TN = TN.item()
    FP = FP.item()
    FN = FN.item()
    TP = TP.item()

    return statistics_from_cm(TN,FN,TP,FP)

def statistics_from_cm(TN,FN,TP,FP):

    # Sensitivity, hit rate, recall, or true positive rate
    try:
        SE = TP/(TP+FN)
    except:
        SE = 0

    # Specificity or true negative rate
    try:
        SP = TN/(TN+FP) 
    except:
        SP = 0

    # Precision or positive predictive value
    try:
        PPV = TP/(TP+FP)
    except:
        PPV = 0

    # Negative predictive value
    try:
        NPV = TN/(TN+FN)
    except:
        NPV = 0

    # Correct classification rate
    CCR = (SE + SP)/2

    return {'CCR': CCR, 'SP':SP, 'SE':SE ,'PPV': PPV, 'NPV': NPV, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP }


def continuous_statistics(actual, predicted, to_string = False):
     r2 = metrics.r2_score(actual, predicted).item()
     explained_variance = metrics.explained_variance_score(actual, predicted).item()
     max_error = metrics.max_error(actual, predicted).item()
     return {"r2":r2, "explained_variance":explained_variance,"max_error":max_error}


def ad_coverage(train_mols, test_mols, descriptor_function, k = 1, z = 1):

    train_descriptors, _ = get_descriptors(train_mols, descriptor_function)
    test_descriptors, _ = get_descriptors(test_mols, descriptor_function)

    predictor = NearestNeighbors(n_neighbors = k + 1, algorithm = "ball_tree")
    predictor.fit(train_descriptors)
    distances, indices = predictor.kneighbors(train_descriptors)
    distances = distances[:,1:] #trim out self distance

    d = np.mean(distances)
    s = np.std(distances)
    D = d + (z*s) #define cutoff

    distances, indices = predictor.kneighbors(test_descriptors)
    distances = distances[:,:-1]
    if distances.shape[1] > 1:
        distances = np.mean(distances, axis = 0)

    accepted = np.where(distances < D, 1, 0)

    coverage = np.sum(accepted) / len(accepted)

    return coverage


#returns k (x, y) tuples in a list
#last set will be largest by at most k-1 if not cleanly divisible
def k_fold_split(l, k = 5):

    np.random.seed(512)

    l = np.array(l)

    np.random.shuffle(l)

    return_list = []

    step = int((1 / k) * len(l))
    for i in range(1,k + 1):
        start = (i - 1) * step
        if i == k:
            fold_l = l[start:]
            return_list.append(fold_l)

        else:
            stop = i * step
            fold_l = l[start:stop]
            return_list.append(fold_l)

    return return_list

class train_test_split:

    def __init__(self, mols, k = 5, verbose = False):
        self.folds = k_fold_split(mols)
        self.k = k
        self.fold_num = 0
        self.verbose = verbose

    def __iter__(self):
        return self

    def __next__(self):

        if self.fold_num >= self.k:
            raise StopIteration()

        train = []
        for i in range(self.k):

            if i != self.fold_num:
                m = self.folds[i]
                train.append(m)

            else:
                test = self.folds[i]

        self.fold_num = self.fold_num + 1

        train = np.concatenate(train)

        if(self.verbose):
            print(f"Train shape: {train_x.shape}")
            print(f"Test shape: {test_x.shape}")

        return (train, test)

def get_morgan_descriptor(mol, radius = 2, rdkit_mol = False):

    if not rdkit_mol:
        mol = mol.mol

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius), morgan)

    return morgan

def get_mudra_paper_descriptor(mol, radius = 2, rdkit_mol = False):

    if not rdkit_mol:
        mol = mol.mol

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius), morgan)

    atom_pair = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol), atom_pair)

    torsion = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol), torsion)

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), maccs)

    return (morgan, atom_pair, torsion, maccs)

def get_maccs_key(mol, rdkit_mol = False):

    if not rdkit_mol:
        mol = mol.mol

    maccs = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), maccs)

    return maccs

#identifying_column_names will be used as keys, everything else will be averaged
def average_results(df, identifying_column_names):

    for_std = ["CCR", "Coverage", "PPV", "NPV", "PPV", "SE", "SP"]

    #TODO: reorder df so identifying_column_names come first
    print(identifying_column_names)

    averaged_results = pd.DataFrame()
    consensus_results = pd.DataFrame()

    possibilities = []
    for column_name in identifying_column_names:
        vals = list(df[column_name].unique())
        full_vals = [(column_name, val) for val in vals]
        possibilities.append(full_vals)


    combs = list(product(*possibilities))

    for comb in combs:

        temp_df = df
        for key,val in comb:
            a = temp_df[temp_df[key] == val]
            temp_df = temp_df[temp_df[key] == val]


        if not temp_df.empty:

            print(temp_df)
            num_averaged = temp_df.shape[0]
            averaged = temp_df.mean(axis = 0)
            std = temp_df.std(axis = 0)
            for key,val in comb:
                averaged[key] = val
            averaged["Num Averaged"] = num_averaged

            for col_name in for_std:
                std = temp_df[col_name].std()
                std_name = f"{col_name}_std"
                averaged[std_name] = std

            fn = np.sum(temp_df["FN"])
            tn = np.sum(temp_df["TN"])
            fp = np.sum(temp_df["FP"])
            tp = np.sum(temp_df["TP"])

            pos = fn + tp
            neg = fp + tn

            cv_stats = statistics_from_cm(tn,fn,tp,fp)

            consensus = pd.Series()
            consensus["Model Name"] = averaged["Model Name"] + " (5-Fold CV)"
            consensus["Descriptor Function"] = averaged["Descriptor Function"]
            for key,val in cv_stats.items():
                consensus[key] = val
            consensus["# Positive Examples"] = neg
            consensus["# Negative Examples"] = pos
            consensus_results = consensus_results.append(consensus, ignore_index = True)


            averaged["Model Name"] = averaged["Model Name"] + " (Averaged)"
            averaged_results = averaged_results.append(averaged, ignore_index = True)

    averaged_results = averaged_results.drop("Fold", axis = 1)
    return averaged_results, consensus_results

def get_all_rdkit_descriptors(mol, rdkit_mol = False):

    if not rdkit_mol:
        mol = mol.mol

    descriptors = np.array([Crippen.MolLogP(mol),
                                      Crippen.MolMR(mol),
                                      Descriptors.FpDensityMorgan1(mol),
                                      Descriptors.FpDensityMorgan2(mol),
                                      Descriptors.FpDensityMorgan3(mol),
                                      Descriptors.FractionCSP3(mol),
                                      Descriptors.HeavyAtomMolWt(mol),
                                      #Descriptors.MaxAbsPartialCharge(mol),
                                      #Descriptors.MaxPartialCharge(mol),
                                      #Descriptors.MinAbsPartialCharge(mol),
                                      #Descriptors.MinPartialCharge(mol),
                                      Descriptors.MolWt(mol),
                                      Descriptors.NumRadicalElectrons(mol),
                                      Descriptors.NumValenceElectrons(mol),
                                      EState.EState.MaxAbsEStateIndex(mol),
                                      EState.EState.MaxEStateIndex(mol),
                                      EState.EState.MinAbsEStateIndex(mol),
                                      EState.EState.MinEStateIndex(mol),
                                      EState.EState_VSA.EState_VSA1(mol),
                                      EState.EState_VSA.EState_VSA10(mol),
                                      EState.EState_VSA.EState_VSA11(mol),
                                      EState.EState_VSA.EState_VSA2(mol),
                                      EState.EState_VSA.EState_VSA3(mol),
                                      EState.EState_VSA.EState_VSA4(mol),
                                      EState.EState_VSA.EState_VSA5(mol),
                                      EState.EState_VSA.EState_VSA6(mol),
                                      EState.EState_VSA.EState_VSA7(mol),
                                      EState.EState_VSA.EState_VSA8(mol),
                                      EState.EState_VSA.EState_VSA9(mol),
                                      Fragments.fr_Al_COO(mol),
                                      Fragments.fr_Al_OH(mol),
                                      Fragments.fr_Al_OH_noTert(mol),
                                      Fragments.fr_aldehyde(mol),
                                      Fragments.fr_alkyl_carbamate(mol),
                                      Fragments.fr_alkyl_halide(mol),
                                      Fragments.fr_allylic_oxid(mol),
                                      Fragments.fr_amide(mol),
                                      Fragments.fr_amidine(mol),
                                      Fragments.fr_aniline(mol),
                                      Fragments.fr_Ar_COO(mol),
                                      Fragments.fr_Ar_N(mol),
                                      Fragments.fr_Ar_NH(mol),
                                      Fragments.fr_Ar_OH(mol),
                                      Fragments.fr_ArN(mol),
                                      Fragments.fr_aryl_methyl(mol),
                                      Fragments.fr_azide(mol),
                                      Fragments.fr_azo(mol),
                                      Fragments.fr_barbitur(mol),
                                      Fragments.fr_benzene(mol),
                                      Fragments.fr_benzodiazepine(mol),
                                      Fragments.fr_bicyclic(mol),
                                      Fragments.fr_C_O(mol),
                                      Fragments.fr_C_O_noCOO(mol),
                                      Fragments.fr_C_S(mol),
                                      Fragments.fr_COO(mol),
                                      Fragments.fr_COO2(mol),
                                      Fragments.fr_diazo(mol),
                                      Fragments.fr_dihydropyridine(mol),
                                      Fragments.fr_epoxide(mol),
                                      Fragments.fr_ester(mol),
                                      Fragments.fr_ether(mol),
                                      Fragments.fr_furan(mol),
                                      Fragments.fr_guanido(mol),
                                      Fragments.fr_halogen(mol),
                                      Fragments.fr_hdrzine(mol),
                                      Fragments.fr_hdrzone(mol),
                                      Fragments.fr_HOCCN(mol),
                                      Fragments.fr_imidazole(mol),
                                      Fragments.fr_imide(mol),
                                      Fragments.fr_Imine(mol),
                                      Fragments.fr_isocyan(mol),
                                      Fragments.fr_isothiocyan(mol),
                                      Fragments.fr_ketone(mol),
                                      Fragments.fr_ketone_Topliss(mol),
                                      Fragments.fr_lactam(mol),
                                      Fragments.fr_lactone(mol),
                                      Fragments.fr_methoxy(mol),
                                      Fragments.fr_morpholine(mol),
                                      Fragments.fr_N_O(mol),
                                      Fragments.fr_Ndealkylation1(mol),
                                      Fragments.fr_Ndealkylation2(mol),
                                      Fragments.fr_NH0(mol),
                                      Fragments.fr_NH1(mol),
                                      Fragments.fr_NH2(mol),
                                      Fragments.fr_Nhpyrrole(mol),
                                      Fragments.fr_nitrile(mol),
                                      Fragments.fr_nitro(mol),
                                      Fragments.fr_nitro_arom(mol),
                                      Fragments.fr_nitro_arom_nonortho(mol),
                                      Fragments.fr_nitroso(mol),
                                      Fragments.fr_oxazole(mol),
                                      Fragments.fr_oxime(mol),
                                      Fragments.fr_para_hydroxylation(mol),
                                      Fragments.fr_phenol(mol),
                                      Fragments.fr_phenol_noOrthoHbond(mol),
                                      Fragments.fr_phos_acid(mol),
                                      Fragments.fr_phos_ester(mol),
                                      Fragments.fr_piperdine(mol),
                                      Fragments.fr_piperzine(mol),
                                      Fragments.fr_priamide(mol),
                                      Fragments.fr_prisulfonamd(mol),
                                      Fragments.fr_pyridine(mol),
                                      Fragments.fr_quatN(mol),
                                      Fragments.fr_SH(mol),
                                      Fragments.fr_sulfide(mol),
                                      Fragments.fr_sulfonamd(mol),
                                      Fragments.fr_sulfone(mol),
                                      Fragments.fr_term_acetylene(mol),
                                      Fragments.fr_tetrazole(mol),
                                      Fragments.fr_thiazole(mol),
                                      Fragments.fr_thiocyan(mol),
                                      Fragments.fr_thiophene(mol),
                                      Fragments.fr_unbrch_alkane(mol),
                                      Fragments.fr_urea(mol),
                                      GraphDescriptors.BalabanJ(mol),
                                      GraphDescriptors.BertzCT(mol),
                                      GraphDescriptors.Chi0(mol),
                                      GraphDescriptors.Chi0n(mol),
                                      GraphDescriptors.Chi0v(mol),
                                      GraphDescriptors.Chi1(mol),
                                      GraphDescriptors.Chi1n(mol),
                                      GraphDescriptors.Chi1v(mol),
                                      GraphDescriptors.Chi2n(mol),
                                      GraphDescriptors.Chi2v(mol),
                                      GraphDescriptors.Chi3n(mol),
                                      GraphDescriptors.Chi3v(mol),
                                      GraphDescriptors.Chi4n(mol),
                                      GraphDescriptors.Chi4v(mol),
                                      GraphDescriptors.HallKierAlpha(mol),
                                      GraphDescriptors.Ipc(mol),
                                      GraphDescriptors.Kappa1(mol),
                                      GraphDescriptors.Kappa2(mol),
                                      GraphDescriptors.Kappa3(mol),
                                      Lipinski.HeavyAtomCount(mol),
                                      Lipinski.NHOHCount(mol),
                                      Lipinski.NOCount(mol),
                                      Lipinski.NumAliphaticCarbocycles(mol),
                                      Lipinski.NumAliphaticHeterocycles(mol),
                                      Lipinski.NumAliphaticRings(mol),
                                      Lipinski.NumAromaticCarbocycles(mol),
                                      Lipinski.NumAromaticHeterocycles(mol),
                                      Lipinski.NumAromaticRings(mol),
                                      Lipinski.NumHAcceptors(mol),
                                      Lipinski.NumHDonors(mol),
                                      Lipinski.NumHeteroatoms(mol),
                                      Lipinski.NumRotatableBonds(mol),
                                      Lipinski.NumSaturatedCarbocycles(mol),
                                      Lipinski.NumSaturatedHeterocycles(mol),
                                      Lipinski.NumSaturatedRings(mol),
                                      Lipinski.RingCount(mol),
                                      MolSurf.LabuteASA(mol),
                                      MolSurf.PEOE_VSA1(mol),
                                      MolSurf.PEOE_VSA10(mol),
                                      MolSurf.PEOE_VSA11(mol),
                                      MolSurf.PEOE_VSA12(mol),
                                      MolSurf.PEOE_VSA13(mol),
                                      MolSurf.PEOE_VSA14(mol),
                                      MolSurf.PEOE_VSA2(mol),
                                      MolSurf.PEOE_VSA3(mol),
                                      MolSurf.PEOE_VSA4(mol),
                                      MolSurf.PEOE_VSA5(mol),
                                      MolSurf.PEOE_VSA6(mol),
                                      MolSurf.PEOE_VSA7(mol),
                                      MolSurf.PEOE_VSA8(mol),
                                      MolSurf.PEOE_VSA9(mol),
                                      MolSurf.SlogP_VSA1(mol),
                                      MolSurf.SlogP_VSA10(mol),
                                      MolSurf.SlogP_VSA11(mol),
                                      MolSurf.SlogP_VSA12(mol),
                                      MolSurf.SlogP_VSA2(mol),
                                      MolSurf.SlogP_VSA3(mol),
                                      MolSurf.SlogP_VSA4(mol),
                                      MolSurf.SlogP_VSA5(mol),
                                      MolSurf.SlogP_VSA6(mol),
                                      MolSurf.SlogP_VSA7(mol),
                                      MolSurf.SlogP_VSA8(mol),
                                      MolSurf.SlogP_VSA9(mol),
                                      MolSurf.SMR_VSA1(mol),
                                      MolSurf.SMR_VSA10(mol),
                                      MolSurf.SMR_VSA2(mol),
                                      MolSurf.SMR_VSA3(mol),
                                      MolSurf.SMR_VSA4(mol),
                                      MolSurf.SMR_VSA5(mol),
                                      MolSurf.SMR_VSA6(mol),
                                      MolSurf.SMR_VSA7(mol),
                                      MolSurf.SMR_VSA8(mol),
                                      MolSurf.SMR_VSA9(mol),
                                      MolSurf.TPSA(mol)], dtype = np.float32)

    descriptors = np.nan_to_num(descriptors, posinf = 0)


    return descriptors

def check_if_mols_in_set(check_mols, set_mols):

    return [c in set_mols for c in check_mols]

def np_array_to_csv(arr):
    s = ""
    for i,val in enumerate(arr):
        if i == 0:
            s = s + str(val)
        else:
            s = s + "," + str(val)

    return s

def sdf_to_fingerprints(filename, score_column_name, descriptor_function,
        output_filename, skip_first_mol = False, limit = -1):

    sdm = SDMolSupplier(filename)
    of = open(output_filename, 'w+')

    for i, mol in enumerate(sdm):

        if skip_first_mol and i == 0:
            continue

        if limit > 0 and i >= limit:
            break

        m = Mol(mol, float(mol.GetProp(score_column_name)))
        fp = descriptor_function(m.mol)
        of.write(f"{np_array_to_csv(fp)}, {m.score}\n")
        print(f"{i} mols fingerprinted\r", end = "")

def read_fingerprint_file(filename):

    fps = []
    scores = []
    f = open(filename, 'r')
    for i, line in enumerate(f):
        s = line.split(',')
        score = s[-1]
        fp = s[:-1]

        fp = np.array(fp, dtype = int)
        score = float(score)
        fps.append(fp)
        scores.append(score)
        print(f"{i} fingerprints read\r", end="")

    return fps, scores

class FingerprintDataset(Dataset):

    def __init__(self, filename):

        self.f = open(filename, 'r')

        #assume each line is the same length
        linelength = 0
        while(True):
            char = self.f.read(1)
            if char == '\n':
                break
            linelength += 1
        #self.line_offset = linelength + 2
        self.line_offset = linelength + 1

        self.f.seek(0,0)
        num_lines = sum([1 for line in self.f])
        self.num_lines = num_lines

    def __getitem__(self, i):
        if type(i) == slice:
            start = i.start
            stop = i.stop
            self.f.seek((start * self.line_offset), 0)
            counter = start
            return_list = []
            while(counter < stop):
                line = self.f.readline()
                s = line.split(',')
                score = s[-1]
                fp = s[:-1]
                fp=np.array(fp, dtype =int)
                score = float(score)
                tup = (fp, score)
                return_list.append(tup)
                counter += 1

            return return_list
        elif type(i) == int:
            self.f.seek((i * self.line_offset), 0)
            line = self.f.readline()
            s = line.split(',')
            score = s[-1]
            fp = s[:-1]
            fp=np.array(fp, dtype =int)
            score = float(score)
        else:
            raise Exception(f"Value passed to getitem ({i}) not compatible")

        return fp, score

    def  __len__(self):

        return self.num_lines

def evaluate_models(mols, dirname, target_name, exclude_models = None):

    from models import RF, SVM, GradientBoosting, NeuralNetwork, GraphCNN, MUDRA
    timestamp = int(time.time())

    dirname_with_time = f"{dirname}/{timestamp}"

    model_dir = f"{dirname_with_time}/models"
    fold_dir = f"{dirname_with_time}/folds"
    stat_dir = f"{dirname_with_time}/statistics"
    plot_dir = f"{dirname_with_time}/plots"

    try:
        os.mkdir(dirname)
    except:
        pass

    try:
        os.mkdir(dirname_with_time)
    except:
        pass

    try:
        os.mkdir(model_dir)
    except:
        pass

    try:
        os.mkdir(fold_dir)
    except:
        pass

    try:
        os.mkdir(stat_dir)
    except:
        pass

    try:
        os.mkdir(plot_dir)
    except:
        pass

    print(f"Saving models and results to '{dirname_with_time}'")

    results = pd.DataFrame()

    for foldnum, data in enumerate(train_test_split(mols)):
        train = data[0]
        test = data[1]

        write_sdf(test, filename = f"{fold_dir}/test_for_fold_{foldnum}.sdf")

        test_scores = [m.score for m in test]

        for descriptor_function, needs_normalization in [(get_all_rdkit_descriptors, True), (get_morgan_descriptor, False), (get_mudra_paper_descriptor, False)]:

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if descriptor_function == get_all_rdkit_descriptors:
                continue
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            coverage = ad_coverage(train, test, descriptor_function, k = 1)

            if descriptor_function == get_mudra_paper_descriptor:
                models = [("MUDRA", MUDRA("classification", descriptor_function = descriptor_function, normalize = needs_normalization))]
            else:
                models = []
                models.append(("Random Forest", RF("classification", descriptor_function = descriptor_function, normalize = needs_normalization, n_estimators = 1000, max_depth = 5)))
                models.append(("SVM", SVM("classification", descriptor_function = descriptor_function, normalize = needs_normalization, gamma = "scale", C = 0.8)))
                models.append(("Neural Network", NeuralNetwork("classification", descriptor_function = descriptor_function, normalize = needs_normalization)))
                models.append(("Gradient Boosting", GradientBoosting("classification", descriptor_function = descriptor_function, normalize = needs_normalization)))
                try: #MUDRA might not be compatible with descriptor
                    models.append(("MUDRA", MUDRA("classification", descriptor_function = descriptor_function, normalize = needs_normalization)))
                except:
                    pass

            for model_name, model in models:

                if exclude_models:
                    if model_name in exclude_models:
                        continue

                model_filename = f"{model_dir}/{target_name.lower()}_{model_name.replace(' ','').lower()}_{descriptor_function.__name__.replace('_','')}_{foldnum}.pkl"
                print(model_filename)
                s = {}
                s["Target"] = target_name
                s["Model Name"] = model_name
                s["Descriptor Function"] = descriptor_function.__name__
                s["Fold"] = foldnum
                s["Train Size"] = len(train)
                s["Test_Size"] = len(test)
                s["Coverage"] = coverage

                model.fit(train)
                pred = model.predict(test)
                try:
                    model_results = classification_statistics(test_scores, pred)
                except:
                    model_results = {}

                print(pred)

                pred_actives = []
                pred_inactives = []
                for i in range(len(pred)):
                    val = pred[i]
                    if val == 1:
                        pred_actives.append(test[i])
                    elif val == 0:
                        pred_inactives.append(test[i])
                    else:
                        raise Exception("Prediction isn't 1 or 0")

                print(len(pred_actives))
                print(len(pred_inactives))

                s.update(model_results)

                active_active = tanimoto_distribution(descriptor_function, pred_actives)
                inactive_inactive = tanimoto_distribution(descriptor_function, pred_inactives)
                active_inactive = tanimoto_distribution(descriptor_function, pred_actives, pred_inactives)

                plot_hist(active_active, "Tanimoto Distribution (Predicted Active vs. Predicted Active)", f"{plot_dir}/tanimoto_active_active_{model_name.replace(' ','').lower()}_fold_{foldnum}.png", xlim = (0,1))
                plot_hist(inactive_inactive, "Tanimoto Distribution (Predicted Inactive vs. Predicted Inactive)", f"{plot_dir}/tanimoto_inactive_inactive_{model_name.replace(' ','').lower()}_fold_{foldnum}.png", xlim = (0,1))
                plot_hist(active_active, "Tanimoto Distribution (Predicted Active vs. Predicted Inactive)", f"{plot_dir}/tanimoto_active_inactive_{model_name.replace(' ','').lower()}_fold_{foldnum}.png", xlim = (0,1))

                s["Active-Active Mean"] = np.mean(active_active)
                s["Active-Active StdDev"] = np.std(active_active)

                s["Active-Inactive Mean"] = np.mean(active_inactive)
                s["Active-Inactive StdDev"] = np.std(active_inactive)

                s["Inactive-Inactive Mean"] = np.mean(inactive_inactive)
                s["Inactive-Inactive StdDev"] = np.std(inactive_inactive)

                s = pd.Series(s)

                print(s)
                results = results.append(s, ignore_index = True)

                model.save(model_filename)

        model_name = "Graph CNN"
        if not (exclude_models and model_name in exclude_models):

            model = GraphCNN("classification", dir_label = "gcnn_output")
            model.fit(train)
            pred = model.predict(test)

            model_results = classification_statistics(test_scores, pred)

            s = {}
            s["Target"] = target_name
            s["Model Name"] = model_name
            s["Fold"] = foldnum
            s["Train Size"] = len(train)
            s["Test_Size"] = len(test)
            s["Coverage"] = -1
            s["Descriptor Function"] = "None"

            active_active = tanimoto_distribution(descriptor_function, pred_actives)
            inactive_inactive = tanimoto_distribution(descriptor_function, pred_inactives)
            active_inactive = tanimoto_distribution(descriptor_function, pred_actives, pred_inactives)

            s["Active-Active Mean"] = np.mean(active_active)
            s["Active-Active StdDev"] = np.std(active_active)

            s["Active-Inactive Mean"] = np.mean(active_inactive)
            s["Active-Inactive StdDev"] = np.std(active_inactive)

            s["Inactive-Inactive Mean"] = np.mean(inactive_inactive)
            s["Inactive-Inactive StdDev"] = np.std(inactive_inactive)

            model_filename = f"{model_dir}/{target_name.lower()}_{model_name.replace(' ','').lower()}_{foldnum}.pkl"
            print(model_filename)

            s.update(model_results)
            s = pd.Series(s)
            print(s)
            results = results.append(s, ignore_index = True)
            model.save(model_filename)

    results.to_csv(f"{stat_dir}/all_modeling_results.csv")

    print(results.columns)
    averaged_results, consensus_results = average_results(results, ["Target", "Model Name", "Descriptor Function"])
    averaged_results.to_csv(f"{stat_dir}/all_modeling_results_averaged.csv")
    consensus_results.to_csv(f"{stat_dir}/all_modeling_results_consensus.csv")

def tanimoto_distribution(descriptor_function, pop_a, pop_b = None):

    same_pop = False

    if pop_b == None:
        pop_b = pop_a
        same_pop = True

    similarities = []
    for i, mol_a in enumerate(pop_a):
        for j, mol_b in enumerate(pop_b):

            if same_pop and i > j:
                continue


            sim = mol_a.get_similarity(mol_b, descriptor_function)

            if sim == 1 and same_pop:
                continue

            similarities.append(sim)

    return similarities

def main():
    from rdkit import Chem
    m1 = Mol(Chem.MolFromSmiles("CCCCCC"), 0)
    m2 = Mol(Chem.MolFromSmiles("CNCCCC"), 0)
    m3 = Mol(Chem.MolFromSmiles("CCCCCC[Cl]"), 0)
    m4 = Mol(Chem.MolFromSmiles("CCCCCC[Br]"), 0)

    pop_a = [m1, m2, m3, m4]
    pop_b = [m2, m3, m4]


    sims = tanimoto_distribution(get_morgan_descriptor, pop_a, pop_b)
    print(sims)

    sims = tanimoto_distribution(get_morgan_descriptor, pop_a)
    print(sims)
if __name__ == "__main__":
    main()

