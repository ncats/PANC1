import pandas as pd
import numpy as np
import glob
import sys
#sys.path.append("/home/josh/git/qsar_modeling")
sys.path.append("qsar_modeling")
from mixture import Mixture
from mixture import EugeneSplit
from mixture import CompoundLeaveOneOut
from mixture import PairLeaveOneOut 
from mixture import get_mixture_descriptors
from mixture import concatenate_descriptors
from mixture import concatenate_descriptors_backwards
from mixture import sum_descriptors
from mixture import average_descriptors
from mol import Mol
from utils import get_morgan_descriptor
from utils import classification_statistics
from utils import get_all_rdkit_descriptors
from utils import statistics_from_cm
from utils import DescriptorHolder
from models import Model, RF, GradientBoosting, NeuralNetwork
import pandas as pd
from rdkit import Chem

def get_mixture_from_series(series, structure_dict):


    if "Label" in series:
        label = series["Label"]
    else:
        label = None
    if series["Sample ID 1"] not in structure_dict or \
       series["Sample ID 2"] not in structure_dict:
           print(series)
           raise Exception("You hate to see it")
    mol_a = structure_dict[series["Sample ID 1"]]
    mol_b = structure_dict[series["Sample ID 2"]]

    print(mol_a)
    print(mol_b)
    if mol_a is None or mol_b is None:
        raise Exception("Failed to read structures into RDKit")
    mol_a = Mol(mol_a, score = None, identifier = series["Sample ID 1"])
    mol_b = Mol(mol_b, score = None, identifier = series["Sample ID 2"])
    mix = Mixture(mol_a, mol_b, score = label)
    return mix

def get_final_models():

    model_names = []
    models = []

    for filename in glob.glob("saved_models/*.model"):

        #if "mixture" in filename:
        #    continue

        trimmed_filename = filename.split("/")[1].split(".")[0]

        model = Model.from_file(filename)
        model_names.append(trimmed_filename)
        models.append(model)

    return model_names, models

def run_models_on_set(model_names, models, mixtures_to_predict, output_filename, sirmsholder):

    labels = [m.identifier for m in mixtures_to_predict]
    pred_df = pd.DataFrame(index = labels, columns = model_names, dtype =float)
    for j in range(len(models)):
        model_name = model_names[j]
        model = models[j]
        if "mixture" in model_name:
            model.descriptor_function = sirmsholder.get_mixture_descriptor
        predictions = model.predict(mixtures_to_predict, run_discretize = False)
        pred_df[model_name] = predictions

    pred_df.to_csv(output_filename)

    return pred_df

def main():

    from sklearn.metrics import roc_auc_score

    mixture_a = Mixture(
            Mol(Chem.MolFromSmiles("c1ccccc1"), score = 0, identifier = "AA"),
            Mol(Chem.MolFromSmiles("CCC(=O)O"), score = 0, identifier = "AB"),
            score = 0, identifier = "MIX_A")

    mixture_b = Mixture(
            Mol(Chem.MolFromSmiles("c1ccc(CC)cc1"), score = 0), 
            Mol(Chem.MolFromSmiles("CCCCCCC(=O)O"), score = 0), 
            score = 0, 
            identifier = "MIX_B")

    example_mixtures = [mixture_a, mixture_b]

    model_names, models = get_final_models()
    kept_model_names = []
    kept_models = []
    for i in range(len(model_names)):
        print(model_names[i])
        print(models[i])
        print(models[i].descriptor_handler)


        #"mixture" models use Sirms descriptors, these require external generation for a given dataset
        #contact Eugene Muratov (murik@email.unc.edu) for access
        if "mixture" in model_names[i]:
            continue

        kept_model_names.append(model_names[i])
        kept_models.append(models[i])

    print(kept_model_names)

    #sirmsholder reads externally generated Sirms descriptors, set to None here as they're not used
    #sirmsholder = DescriptorHolder(ncats_backup, filename = "backup_two_combos/adjusted_backup_two_combos_simplex.txt", delimiter = '\t')

    sirmsholder = None

    model_names = kept_model_names
    models = kept_models
    df = run_models_on_set(model_names, models, example_mixtures, "test.csv", sirmsholder)

    print("Finished.")

if __name__ == "__main__":
    main()
