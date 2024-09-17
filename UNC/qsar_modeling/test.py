import numpy as np
import unittest

from mol import Mol
from models import RF, GradientBoosting, NeuralNetwork
from utils import get_morgan_descriptor
from mixture import Mixture
from mixture import concatenate_descriptors

class TestMixture(unittest.TestCase):

    def test_Mixture(self):

        mol1 = Mol.from_smiles("CCCCCC", score = 1, identifier = "mol_a_identifier")
        mol2 = Mol.from_smiles("CCCCCN", score = 0, identifier = "mol_b_identifier")
        mixture = Mixture(mol1, mol2, score = 0)

    def test_Mixture_provide_identifier(self):

        mol1 = Mol.from_smiles("CCCCCC", score = 1)
        mol2 = Mol.from_smiles("CCCCCN", score = 0, identifier = "mol_b_identifier")
        mixture = Mixture(mol1, mol2, score = 0, identifier = "mixture_identifier")


    @unittest.expectedFailure
    def test_fail_Mixture_no_mol_identifier(self):

        mol1 = Mol.from_smiles("CCCCCC", score = 1)
        mol2 = Mol.from_smiles("CCCCCN", score = 0, identifier = "mol_b_identifier")
        mixture = Mixture(mol1, mol2, score = 0)

    @unittest.expectedFailure
    def test_fail_Mixture_None_mol(self):

        mol1 = None
        mol2 = Mol.from_smiles("CCCCCN", score = 0, identifier = "mol_b_identifier")
        mixture = Mixture(mol1, mol2, score = 0)



class TestMol(unittest.TestCase):

    def test_from_smiles(self):
        smiles = "CCCCCC"
        mol = Mol.from_smiles(smiles, score = 1)
        mol = Mol.from_smiles(smiles, score = None)

class TestCompoundModels(unittest.TestCase):

    def setUp(self):

        mol1 = Mol.from_smiles("CCCCCC", score = 0)
        mol2 = Mol.from_smiles("CCCCCN", score = 1)

        mols = [mol1, mol2]
        self.mols = mols

    def test_nothing(self):
        assert(True == True)

    def test_classification_RF(self):
        model = RF(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Single", descriptor_handler = None, batch_size = None, normalize = False, y_randomize = False, n_estimators = 100)

        model.fit(self.mols)
        pred = model.predict(self.mols)

        actual_scores = np.array([m.score for m in self.mols])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)

    def test_classification_GradientBoosting(self):
        model = GradientBoosting(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Single", descriptor_handler = None, batch_size = None, normalize = False, y_randomize = False)

        model.fit(self.mols)
        pred = model.predict(self.mols)

        actual_scores = np.array([m.score for m in self.mols])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)

    def test_classification_NeuralNetwork(self):
        model = NeuralNetwork(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Single", descriptor_handler = None, batch_size = None, normalize = False, y_randomize = False, verbose = True)

        model.fit(self.mols)
        pred = model.predict(self.mols)

        actual_scores = np.array([m.score for m in self.mols])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)

class TestMixtureModels(unittest.TestCase):

    def setUp(self):

        mol1 = Mol.from_smiles("CCCCCC", score = 0)
        mol2 = Mol.from_smiles("CCCCCN", score = 1)
        mixture1 = Mixture(mol1, mol2, identifier = "mixture_1", score = 0)

        mol1 = Mol.from_smiles("CCCCCC", score = 0)
        mol2 = Mol.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", score = 0)
        mixture2 = Mixture(mol1, mol2, identifier = "mixture_2", score = 1)

        self.mixtures = [mixture1, mixture2]

    def test_classification_RF(self):
        model = RF(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Mixture", descriptor_handler = concatenate_descriptors, batch_size = None, normalize = False, y_randomize = False, n_estimators = 100)

        model.fit(self.mixtures)
        pred = model.predict(self.mixtures)

        actual_scores = np.array([m.score for m in self.mixtures])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)

    def test_classification_GradientBoosting(self):
        model = GradientBoosting(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Mixture", descriptor_handler = concatenate_descriptors, batch_size = None, normalize = False, y_randomize = False, n_estimators = 100)

        model.fit(self.mixtures)
        pred = model.predict(self.mixtures)

        actual_scores = np.array([m.score for m in self.mixtures])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)

    def test_classification_NeuralNetwork(self):
        model = GradientBoosting(task = "classification", descriptor_function = get_morgan_descriptor, data_type = "Mixture", descriptor_handler = concatenate_descriptors, batch_size = None, normalize = False, y_randomize = False, n_estimators = 100)

        model.fit(self.mixtures)
        pred = model.predict(self.mixtures)

        actual_scores = np.array([m.score for m in self.mixtures])

        match = np.all(np.equal(actual_scores, pred))
        assert(match)



'''
from run_models import read_sdf
from run_models import classification_statistics
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem

def get_custom_descriptor(mol):


    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,
        radius = 3, nBits = 2048), morgan)

    return morgan

def get_fun_descriptor(mol):

    morgan = np.array((0,))
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol,
        radius = 4, nBits = 2048), morgan)

    return morgan


def main():

    from models import SVM, RF, GradientBoosting, GraphCNN
    from mol import Mol
    m = SVM("classification", descriptor_function = get_custom_descriptor, gamma = "scale")

    from rdkit.Chem import MolFromSmiles

    mol1 = MolFromSmiles("CCCCCC")
    mol2 = MolFromSmiles("CCNCCC")

    mol_obj_1 = Mol(mol1, 4)
    mol_obj_2 = Mol(mol2, 6)
    mols = [mol_obj_1, mol_obj_2]
    m.fit(mols)
    print("Reading in data...")

    train_mols, train_scores, _, _ = read_sdf("/data/curated/3D7/dataset_3D7.sdf",
            score_column_name = "Outcome",pos_class_name = "Active",
            neg_class_name = "Inactive")

    mols = []
    for mol, score in zip(train_mols, train_scores):
        m = Mol(mol, score)
        mols.append(m)

    model = GraphCNN("classification", dir_label = "gcnn_test")
    model.fit(mols)
    pred = model.predict(mols)
    print(pred)
    exit()
    pred = model.predict(mols)
    print(classification_statistics(train_scores, pred))

    model = RF("classification", descriptor_function = get_custom_descriptor, n_estimators = 1000)
    model.fit(mols)
    pred = model.predict(mols)
    print(classification_statistics(train_scores, pred))

    model = SVM("classification", descriptor_function = get_custom_descriptor, gamma = "scale")
    model.fit(mols)
    pred = model.predict(mols)
    print(classification_statistics(train_scores, pred))

    model = GradientBoosting("classification", descriptor_function = get_custom_descriptor)
    model.fit(mols)
    pred = model.predict(mols)
    print(classification_statistics(train_scores, pred))



    exit()

    test_mols, test_scores = read_sdf("/data/curated/betalactamase/betalactamase_ext_data.sdf",
            score_column_name = "Outcome",pos_class_name = "1",
            neg_class_name = "0")

    test_descriptors = get_descriptors(test_mols, descriptor_function =
            get_custom_descriptor)
    test_descriptors = torch.tensor(test_descriptors,  dtype=torch.float).cuda()

    models, statistics = evaluate_models(train_mols, train_scores, objective =
            "classification", k = 5, descriptor_function = get_custom_descriptor)

    print(statistics)

    model_list = models["Neural Network"]
    preds = []
    for model in model_list:
        pred = model.forward(test_descriptors)
        preds.append(pred)

    preds = torch.stack(preds, dim = 1)
    consensus_pred = torch.mean(preds, dim = 1)
    print(preds.shape)

    consensus_pred = discretize(consensus_pred.cpu())
    s = classification_statistics(test_scores, consensus_pred)
    print(s)

'''
if __name__ == "__main__":
    unittest.main()
