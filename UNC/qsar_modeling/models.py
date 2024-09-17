import numpy as np

from rdkit import Chem

from mol import get_descriptors
from mixture import get_mixture_descriptors

from normalize import Normalizer

#GraphCNN
import sys
sys.path.insert(0, '/home/josh/git/my_openchem/')
from openchem.models.Graph2Label import Graph2Label
from openchem.modules.encoders.gcn_encoder import GraphCNNEncoder
from openchem.modules.mlp.openchem_mlp import OpenChemMLP
from openchem.data.graph_data_layer import GraphDataset
from openchem.data.utils import create_loader
from openchem.utils.graph import Attribute, Graph
from openchem.utils.utils import identity
from openchem.models.openchem_model import build_training, fit, evaluate
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import RMSprop, SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.nn.parallel import DataParallel

import os
sys.path.insert(0, '/home/josh/git/pyMuDRA/')
from pymudra.mudra import MUDRAEstimator

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

import pickle

class Model:

    def __init__(self, task, descriptor_function, data_type = "Single", batch_size = None, normalize = False, descriptor_handler = None, y_randomize = False):

        self.task = task

        '''
        if not descriptor_function:
            raise Exception("Must provide a function to generate descriptors from RDKit ROMol object")
        '''

        if data_type == "Single":
            if descriptor_handler != None:
                raise Exception("Descriptor handlers currently only supported for 'Mixture' mode")

        if data_type not in ["Single", "Mixture"]:
            raise Exception(f"data_type must be one of 'Single' or 'Mixture' ({data_type} provided)")

        self.data_type = data_type

        self.descriptor_function = descriptor_function
        self.descriptor_handler = descriptor_handler

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = 1024

        self.normalize = normalize
        self.y_randomize = y_randomize

    def _get_descriptors(self, inputs):
        if self.data_type == "Mixture":
            if self.descriptor_handler != None:
                descriptors = [m.get_descriptor_with_handler(self.descriptor_handler, self.descriptor_function) for m in inputs]
            else:
                descriptors = [m.get_descriptor(self.descriptor_function) for m in inputs]
            scores = [m.score for m in inputs]
        elif self.data_type == "Single":
            descriptors, scores = get_descriptors(inputs, self.descriptor_function)

        '''
        descriptors, scores = get_descriptors(inputs, self.descriptor_function)
        '''



        return descriptors, scores

    def fit(self, mols):

        descriptors, scores = self._get_descriptors(mols, )
        descriptors = np.array(descriptors)
        scores = np.array(scores)

        if self.y_randomize:
            np.random.shuffle(scores)

        if self.normalize:
            self.normalizer = Normalizer()
            self.normalizer.fit(descriptors, scores)
            descriptors = self.normalizer.transform(descriptors)

        train_x = descriptors
        train_y = scores

        self._specific_fit(train_x, train_y)

    def predict(self, mols, run_discretize = True):

        if not self.model:
            raise Exception("'predict' called before 'fit'")

        descriptors, _ = self._get_descriptors(mols)
        descriptors = np.array(descriptors)

        if self.normalize:
            descriptors = self.normalizer.transform(descriptors)

        test_x = np.array(descriptors)
        preds = self._specific_predict(test_x)

        return_type = float
        if run_discretize and self.task == "classification":
            preds = discretize(preds)
            return_type = int

        return np.array(preds, dtype = return_type)

    def save(self, filename):

        pickle.dump(self, open(filename, "wb"))

    @classmethod
    def from_file(cls, filename):

        m = pickle.load(open(filename, "rb"))
        return m

class SVM(Model):

    def __init__(self, task, descriptor_function, batch_size = None, normalize = False, **kwargs):

        from sklearn import svm

        super().__init__(task, descriptor_function, batch_size, normalize)

        self.name = "Support Vector Machine"

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for SVM")

        if self.task == "classification":
            self.model = svm.SVC(**kwargs)
        if self.task == "regression":
            print("SVM regression")
            self.model = svm.SVR(**kwargs)

    def fit(self, mols):

        descriptors, scores = get_descriptors(mols, self.descriptor_function)
        if self.y_randomize:
            np.random.shuffle(scores)

        if self.normalize:
            self.normalizer = Normalizer()
            self.normalizer.fit(descriptors, scores)
            descriptors = self.normalizer.transform(descriptors)

        train_x = np.array(descriptors)
        train_y = np.array(scores)
        self.model.fit(train_x, train_y)

    def predict(self, mols):

        if not self.model:
            raise Exception("'predict' called before 'fit'")

        descriptors, _ = get_descriptors(mols, self.descriptor_function)

        if self.normalize:
            descriptors = self.normalizer.transform(descriptors)

        test_x = np.array(descriptors)

        return self.model.predict(test_x)

class RF(Model):

    def __init__(self, task, descriptor_function, data_type = "Single", descriptor_handler = None, batch_size = None, normalize =
            False, y_randomize = False, n_estimators = 100, **kwargs):

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor

        self.name = "Random Forest"

        super().__init__(task = task, descriptor_function = descriptor_function, data_type = data_type, descriptor_handler = descriptor_handler, batch_size = batch_size, normalize = normalize, y_randomize = y_randomize)

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for RF")

        if self.task == "classification":
            self.model = RandomForestClassifier(n_estimators = n_estimators, **kwargs)
        if self.task == "regression":
            self.model = RandomForestRegressor(n_estimators = n_estimators, **kwargs)

    def _specific_fit(self, x, y):

        self.model.fit(x, y)

    def _specific_predict(self, x):

        if self.task == "classification":
            active_probability = (self.model.predict_proba(x)[:, 1])
            return active_probability
        else:
            return self.model.predict(x)

class GradientBoosting(Model):

    def __init__(self, task, descriptor_function, data_type = "Single", descriptor_handler = None, batch_size = None, normalize = False, y_randomize = False, **kwargs):

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import GradientBoostingRegressor

        super().__init__(task = task, descriptor_function = descriptor_function, data_type = data_type, descriptor_handler = descriptor_handler, batch_size = batch_size, normalize = normalize, y_randomize = y_randomize)

        self.name = "Gradient Boosting"

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for GradientBoosting")

        if self.task == "classification":
            self.model = GradientBoostingClassifier(**kwargs)
        if self.task == "regression":
            self.model = GradientBoostingRegressor(**kwargs)

    def _specific_fit(self, x, y):

        self.model.fit(x, y)

    def _specific_predict(self, x):

        if self.task == "classification":
            active_probability = (self.model.predict_proba(x)[:, 1])
            return active_probability
        else:
            return self.model.predict(x)

class GraphCNN(Model):

    def __init__(self, task, dir_label, batch_size = None):
        super().__init__(task, batch_size = batch_size, descriptor_function = None)

        self.name = "Graph CNN"

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for GraphCNN")

        self.dir_label = dir_label

        node_attributes = {}
        node_attributes['valence'] = Attribute('node', 'valence', one_hot=True, values=[1, 2, 3, 4, 5, 6,7,8,9,10])
        node_attributes['charge'] = Attribute('node', 'charge', one_hot=True, values=[-1, 0, 1, 2, 3, 4])
        node_attributes['hybridization'] = Attribute('node', 'hybridization',
                                                     one_hot=True, values=[0, 1, 2, 3, 4, 5, 6, 7])
        node_attributes['aromatic'] = Attribute('node', 'aromatic', one_hot=True,
                                                values=[0, 1])
        node_attributes['atom_element'] = Attribute('node', 'atom_element',
                                                    one_hot=True,
                                                    values=list(range(11)))
        self.node_attributes = node_attributes
        max_size = 6 #hard code for dummy benzene

        dummy_graph = Graph.from_smiles("CCCCCC", max_size = max_size, get_atom_attributes = self.get_atomic_attributes, get_bond_attributes = None)
        dummy_matrix = dummy_graph.get_node_feature_matrix(all_atr_dict = self.node_attributes, max_size = max_size)
        input_size = dummy_matrix.shape[1]

        if task == "regression":
            criterion = nn.MSELoss()
        elif task == "classification":
            criterion = nn.BCEWithLogitsLoss()


        self.model_object = Graph2Label
        self.model_config = {
            'task': 'regression',
            'data_layer': GraphDataset,
            'use_clip_grad': False,
            'batch_size': 64,
            'num_epochs': 100,
            'print_every': 10,
            'save_every': 5,
            'eval_metrics': metrics.r2_score,
            'criterion': criterion,
            'optimizer': Adam,
            'optimizer_params': {
                'lr': 0.0005,
            },
            'lr_scheduler': StepLR,
            'lr_scheduler_params': {
                'step_size': 15,
                'gamma': 0.8
            },
            'encoder': GraphCNNEncoder,
            'encoder_params': {
                #'input_size': train_dataset.num_features,
                'input_size': input_size,
                'encoder_dim': 128,
                'n_layers': 5,
                'hidden_size': [128, 128, 128, 128, 128],
            },
            'mlp': OpenChemMLP,
            'mlp_params': {
                'input_size': 128,
                'n_layers': 2,
                'hidden_size': [128, 1],
                'activation': [F.relu, identity]
            },
            'world_size':1,
            'random_seed':42,
        }


    def fit(self, mols):


        mols, scores = self.get_mols_and_scores(mols)

        train_dataset = GraphDataset.from_mol_list(self.get_atomic_attributes, self.node_attributes,
                         mols, scores)

        train_loader = create_loader(train_dataset,
                                     batch_size = self.model_config['batch_size'],
                                     shuffle=True,
                                     num_workers=1,
                                     pin_memory=True,
                                     drop_last = True,
                                     sampler=None)

        self.model_config['train_data_layer'] = train_dataset
        self.model_config['val_data_layer'] = train_dataset #won't use validation, just needed for compatibility
        self.model_config['train_loader'] = train_loader
        self.model_config['val_loader'] = train_loader #won't use validation, just needed for compatibility

        logdir = "trained_models"
        try:
            os.stat(logdir)
        except:
            os.mkdir(logdir)

        logdir = logdir + f"/{self.dir_label}"
        try:
            os.stat(logdir)
        except:
            os.mkdir(logdir)

        checkpoint_dir = logdir + "/checkpoint/"
        try:
            os.stat(checkpoint_dir)
        except:
            os.mkdir(checkpoint_dir)

        self.model_config['logdir'] = logdir
        self.model_config['use_cuda'] = True

        model = self.model_object(params=self.model_config)
        model = model.cuda()
        model = DataParallel(model)

        self.criterion, self.optimizer, self.lr_scheduler = build_training(model, self.model_config)
        fit(model, self.lr_scheduler, train_loader, self.optimizer, self.criterion, self.model_config)

        self.model = model

    def predict(self, mols):

        if not self.model:
            raise Exception("'predict' called before 'fit'") 
        mols, scores = self.get_mols_and_scores(mols)

        test_dataset = GraphDataset.from_mol_list(self.get_atomic_attributes, self.node_attributes,
                         mols, scores)

        test_loader = create_loader(test_dataset,
                                     batch_size = self.model_config['batch_size'],
                                     shuffle=False,
                                     num_workers=1,
                                     pin_memory=True,
                                     sampler=None)

        loss, _metrics, prediction = evaluate(self.model, test_loader, self.criterion)


        predictions = np.array(prediction).squeeze()

        if self.task == "classification":
            predictions = discretize(predictions)

        return predictions

    #make static?
    def get_atomic_attributes(self, atom):
        attr_dict = {}

        atomic_num = atom.GetAtomicNum()
        atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8,
                          53: 9}
        if atomic_num in atomic_mapping.keys():
            attr_dict['atom_element'] = atomic_mapping[atomic_num]
        else:
            attr_dict['atom_element'] = 10
        attr_dict['valence'] = atom.GetTotalValence()
        attr_dict['charge'] = atom.GetFormalCharge()
        attr_dict['hybridization'] = atom.GetHybridization().real
        attr_dict['aromatic'] = int(atom.GetIsAromatic())
        return attr_dict

    #make static?
    def get_mols_and_scores(self, mols):

        mol_list = [m.mol for m in mols]
        scores = torch.tensor([m.score for m in mols])

        return mol_list, scores

class DeepDockingNN(nn.Module):

    def __init__(self, vec_length):

        super(VanillaNN, self).__init__()
        self.fc1 = nn.Linear(vec_length, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, 1)

        self.d1 = nn.Dropout(0.7)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d1(x)
        x = F.relu(self.fc3(x))
        x = self.d1(x)
        x = F.relu(self.fc4(x))
        x = self.d1(x)
        return x


class VanillaNN(nn.Module):

    def __init__(self, vec_length, task):
         
        if task not in ["classification", "regression"]:
            raise Exception("VanillaNN class needs a task of 'classification' or 'regression'")

        self.task = task

        super(VanillaNN, self).__init__()
        self.fc1 = nn.Linear(vec_length, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10000)
        self.fc4 = nn.Linear(10000, 10000)
        self.fc5 = nn.Linear(10000, 1000)
        self.fc6 = nn.Linear(1000, 100)
        self.fc7 = nn.Linear(100, 1)
        if self.task == "classification":
            self.sigmoid = nn.Sigmoid()

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.d2(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = torch.sigmoid(x)
        return x

class NeuralNetwork(Model):

    def __init__(self, task, descriptor_function, data_type = None, descriptor_handler = None, batch_size = None, normalize = False, y_randomize = False, verbose = False, **kwargs):

        super().__init__(task, descriptor_function = descriptor_function, data_type = data_type,
                batch_size = batch_size, descriptor_handler = descriptor_handler, normalize = normalize,
                y_randomize = y_randomize)

        self.name = "Neural Network"
        self.verbose = verbose

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for NeuralNetwork")

    def _specific_fit(self, x, y):

        train_x = torch.tensor(x, dtype=torch.float).cuda()
        train_y = torch.tensor(y, dtype=torch.float).cuda()

        train_y = train_y.unsqueeze(dim = 1)

        vec_length = train_x.shape[1]

        model = VanillaNN(vec_length, task = self.task)
        model = model.cuda()
        if self.task == "regression":
            criterion = nn.MSELoss()
        elif self.task == "classification":
            criterion = nn.BCEWithLogitsLoss()

        learning_rate = 0.0001

        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        num_epochs = 100

        if self.verbose:
            print("")

        #train_loop
        for x in range(num_epochs):
            if self.verbose:
                print(f"\rEpoch {x}", end = "")

            pred = model.forward(train_x)
            loss = criterion(pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.verbose:
            print("")

        self.model = model


    def _specific_predict(self, x):

        test_x = torch.tensor(x, dtype=torch.float).cuda()

        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward(test_x).cpu()

        prediction = np.array(prediction.flatten())
        return prediction

    '''
    def save(self, filename):
        if not self.model:
            raise Exception("Can't save model before fitting (model parameters depend on descriptor length)")

        torch.save(filename)

    @classmethod
    def from_file(cls, filename):


        m = torch.load(filename
    '''

class MUDRA(Model):

    def __init__(self, task, descriptor_function, batch_size = None, normalize = False, **kwargs):

        super().__init__(task, descriptor_function, batch_size, normalize)

        self.name = "MUDRA"

        #ensure descriptor function returns at least two distinct descriptors
        m = Chem.MolFromSmiles("CCCCCC")
        desc = descriptor_function(m)
        if not isinstance(desc, tuple):
            raise Exception(f"Descriptor function '{descriptor_function.__name__}()' does not provide at least two descriptors for MUDRA")

        if task not in ["classification", "regression"]:
            raise Exception(f"Modeling task '{task}' not supported for SVM")

        if self.task == "classification":
            self.model = MUDRAEstimator('classifier')
        if self.task == "regression":
            self.model = MUDRAEstimator('regressor')

    def fit(self, mols):

        descriptors, scores = get_descriptors(mols, self.descriptor_function, concatenate = False)

        if self.normalize:
            self.normalizer = Normalizer()
            self.normalizer.fit(descriptors, scores)
            descriptors = self.normalizer.transform(descriptors)

        train_x = [list(i) for i in zip(*descriptors)]
        train_x = [np.array(i) for i in train_x]
        train_y = np.array(scores)
        self.model.fit(train_x, train_y)

    def predict(self, mols):

        if not self.model:
            raise Exception("'predict' called before 'fit'")

        descriptors, _ = get_descriptors(mols, self.descriptor_function, concatenate = False)

        if self.normalize:
            descriptors = self.normalizer.transform(descriptors)

        test_x = [list(i) for i in zip(*descriptors)]
        test_x = [np.array(i) for i in test_x]

        return self.model.predict(test_x)

#values equal to or above cutoff become 1, values below cutoff become 0
def discretize(x, cutoff = 0.5):

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype = float)

    return torch.ge(x, cutoff).squeeze()



