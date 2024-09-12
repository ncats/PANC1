import os, random, math
import copy, sys
import csv
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import trange

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.train import evaluate_predictions
from cancer_train import DiseaseModel
from scipy.stats import spearmanr


def predict(model, data, args):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for i in trange(0, len(data), args.batch_size):
            mol_batch = MoleculeDataset(data[i:i + args.batch_size])
            smiles1, smiles2 = mol_batch.compound_names(), mol_batch.smiles()
            targets = mol_batch.targets()
            preds = model.combo_forward(smiles1, smiles2, mode=0)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets)

    #if len(all_targets) > 0:
    #    res = spearmanr(np.array(all_preds)[:, 0], np.array(all_targets)[:, 0]) 
    #    print(res)
    #    #score = evaluate_predictions(all_preds, all_targets, 1, args.metric_func, args.dataset_type)
    #    #print(f'test auc = {score}', file=sys.stderr)

    return np.array(all_preds)


def run_testing(args):
    checkpoint_paths = []
    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                checkpoint_paths.append(os.path.join(root, fname))

    ckpt = torch.load(checkpoint_paths[0])
    model_args = ckpt['args']
    model_args.use_compound_names = True
    test_data = get_data(path=args.test_path, args=model_args)

    sum_preds = np.zeros((len(test_data), 1))
    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path)
        model_args = ckpt['args']
        model = DiseaseModel(model_args).cuda()
        model.load_state_dict(ckpt['state_dict'])
        test_preds = predict(model, test_data, model_args)
        sum_preds += np.array(test_preds)

    ref_data = {}
    with open(args.ref_path) as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            ref_data[(row[1], row[6])] = row

    sum_preds /= len(checkpoint_paths)
    for x,y,pred in zip(test_data.compound_names(), test_data.smiles(), sum_preds):
        d = ref_data[(x, y)] if (x,y) in ref_data else ["None"]
        print('\t'.join(d) + '\t' + str(pred[0]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ref_path', default='data/cancer/Test_set.csv')
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    args = parser.parse_args()

    run_testing(args)
    
