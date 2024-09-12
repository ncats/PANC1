import os, random, math
import copy
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import trange

from chemprop.parsing import add_train_args, modify_train_args
from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
from chemprop.train import evaluate, evaluate_predictions, predict
from tape.optimization import WarmupLinearSchedule


class DiseaseModel(nn.Module):

    def __init__(self, args):
        super(DiseaseModel, self).__init__()
        self.encoder = MoleculeModel(classification=False, multiclass=False)
        self.encoder.create_encoder(args)
        self.encoder.ffn = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.latent_size)
        )
        self.src_ffn = nn.Linear(args.latent_size, args.num_tasks)
        self.tgt_ffn = nn.Linear(args.latent_size, 1)
        self.ffn = [self.tgt_ffn, self.src_ffn]  # main target first

        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def DTI_forward(self, smiles_batch):
        DTI_vecs = self.encoder(smiles_batch)
        return torch.sigmoid(DTI_vecs)

    def forward(self, smiles_batch, mode):
        DTI_vecs = self.DTI_forward(smiles_batch)
        return self.ffn[mode](DTI_vecs)

    def combo_forward(self, smiles1, smiles2, mode):
        DTI_vecs1 = self.DTI_forward(smiles1)
        DTI_vecs2 = self.DTI_forward(smiles2)
        DTI_vecs = DTI_vecs1 + DTI_vecs2 - DTI_vecs1 * DTI_vecs2
        score1 = torch.sigmoid(self.ffn[mode](DTI_vecs1))
        score2 = torch.sigmoid(self.ffn[mode](DTI_vecs2))
        score = self.ffn[mode](DTI_vecs)
        bliss = torch.log(score1 + score2 - score1 * score2)
        return score - bliss


def prepare_data(args):
    src_data = get_data(path=args.data_path, args=args)
    dti_data = get_data(path=args.dti_path, args=args)
    tgt_data = get_data(path=args.tgt_path, args=args)

    args.use_compound_names = True
    src_combo = get_data(path=args.src_combo, args=args)
    tgt_combo = get_data(path=args.tgt_combo, args=args)
    args.use_compound_names = False

    tgt_combo_train, tgt_combo_val, tgt_combo_test = split_data(data=tgt_combo, split_type='random', sizes=(0.8,0.1,0.1), seed=args.seed)
    print(len(tgt_combo_train), len(tgt_combo_val), len(tgt_combo_test))

    args.output_size = len(dti_data[0].targets)
    args.num_tasks = len(src_data[0].targets)
    args.train_data_size = len(tgt_data)
    return dti_data, src_data, src_combo, tgt_data, tgt_combo_train, tgt_combo_val, tgt_combo_test


def train(dti_data, src_data, src_combo, tgt_data, tgt_combo, model, optimizer, scheduler, loss_func, args):
    model.train()
    src_data.shuffle()
    tgt_data.shuffle()
    dti_data.shuffle()
    src_combo.shuffle()

    tgt_data = [d for d in tgt_data] + [d for d in tgt_data] 
    tgt_data = [d for d in tgt_data] + [d for d in tgt_data]
    src_combo = [d for d in src_combo] + [d for d in src_combo]

    for i in trange(0, len(tgt_data), args.batch_size):
        model.zero_grad()
        tgt_combo.shuffle()  # combo is small, reshuffle everytime

        src_batch = MoleculeDataset(src_data[i:i + args.batch_size])
        tgt_batch = MoleculeDataset(tgt_data[i:i + args.batch_size])
        dti_batch = MoleculeDataset(dti_data[i:i + args.batch_size])
        src_combo_batch = MoleculeDataset(src_combo[i:i + args.batch_size])
        tgt_combo_batch = MoleculeDataset(tgt_combo[:args.batch_size])  # only take the first batch
        if len(tgt_batch) < args.batch_size:
            continue
        
        # DTI batch
        smiles, targets = dti_batch.smiles(), dti_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model.encoder(smiles)[:, :targets.size(1)]
        dti_loss = loss_func(preds, targets)
        dti_loss = (dti_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        smiles, targets = src_batch.smiles(), src_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model(smiles, mode=1)
        src_loss = loss_func(preds, targets)
        src_loss = (src_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        smiles, targets = tgt_batch.smiles(), tgt_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model(smiles, mode=0)
        tgt_loss = loss_func(preds, targets)
        tgt_loss = (tgt_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        smiles1, smiles2 = src_combo_batch.compound_names(), src_combo_batch.smiles()
        targets = src_combo_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model.combo_forward(smiles1, smiles2, mode=1)
        src_combo_loss = loss_func(preds, targets)
        src_combo_loss = (src_combo_loss * mask).sum() / mask.sum()
        smiles1 = smiles2 = targets = mask = None

        smiles1, smiles2 = tgt_combo_batch.compound_names(), tgt_combo_batch.smiles()
        targets = tgt_combo_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model.combo_forward(smiles1, smiles2, mode=0)
        tgt_combo_loss = loss_func(preds, targets)
        tgt_combo_loss = (tgt_combo_loss * mask).sum() / mask.sum()

        loss = args.dti_lambda * dti_loss + args.single_lambda * (src_loss + tgt_loss) + args.combo_lambda * (src_combo_loss + tgt_combo_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()


def combo_evaluate(model, data, args):
    model.eval()
    all_preds, all_targets = [], []

    for i in trange(0, len(data), args.batch_size):
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles1, smiles2 = mol_batch.compound_names(), mol_batch.smiles()
        targets = mol_batch.targets()
        preds = model.combo_forward(smiles1, smiles2, mode=0)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets)

    print([x[0] for x in all_preds])
    score = evaluate_predictions(all_preds, all_targets, 1, args.metric_func, args.dataset_type)
    return score


def run_training(args, save_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dti_data, src_data, src_combo, tgt_data, tgt_combo_train, tgt_combo_val, tgt_combo_test = prepare_data(args)

    model = DiseaseModel(args).cuda()
    loss_func = get_loss_func(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer, 
            warmup_steps=args.train_data_size / args.batch_size * 2,
            t_total=args.train_data_size / args.batch_size * args.epochs
    )

    args.metric_func = get_metric_func(metric=args.metric)
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train(dti_data, src_data, src_combo, tgt_data, tgt_combo_train, model, optimizer, scheduler, loss_func, args)

        dti_preds = predict(model.encoder, dti_data, args.batch_size)
        dti_preds = [pred[:args.output_size] for pred in dti_preds]
        val_scores = evaluate_predictions(dti_preds, dti_data.targets(), args.output_size, args.metric_func, args.dataset_type)
        avg_val_score = np.nanmean(val_scores)
        print(f'DTI Validation {args.metric} = {avg_val_score:.4f}')

        val_scores = combo_evaluate(model, tgt_combo_val, args)
        avg_val_score = np.nanmean(val_scores)
        print(f'Combo Validation {args.metric} = {avg_val_score:.4f}')

        if args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score:
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, args=args)

        # for DTI model visualization (slightly relax)
        if args.minimize_score and avg_val_score < best_score + 0.02 or not args.minimize_score and avg_val_score > best_score - 0.02:
            save_checkpoint(os.path.join(save_dir, 'model.dti'), model, args=args)
            print(f"DTI model saved at {epoch}.")

    print(f'Loading model checkpoint from epoch {best_epoch}')
    ckpt_path = os.path.join(save_dir, 'model.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    test_scores = combo_evaluate(model, tgt_combo_test, args)
    avg_test_scores = np.nanmean(test_scores)
    print(f'Test {args.metric} = {avg_test_scores:.4f}')

    return avg_test_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dti_path', default="data/cancer/dti.csv")
    parser.add_argument('--tgt_path', default="data/cancer/panc1_single.csv")
    parser.add_argument('--src_combo', default="data/cancer/nci_combo.csv")
    parser.add_argument('--tgt_combo', default="data/cancer/panc1_combo.csv")
    parser.add_argument('--single_lambda', type=float, default=1)
    parser.add_argument('--combo_lambda', type=float, default=1)
    parser.add_argument('--dti_lambda', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=100)

    add_train_args(parser)
    args = parser.parse_args()
    args.data_path = 'data/cancer/nci_single.csv'
    args.dataset_type = 'classification'
    args.num_folds = 5

    modify_train_args(args)
    print(args)

    all_test_scores = [0] * args.num_folds
    for i in range(0, args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        args.seed = i
        all_test_scores[i] = run_training(args, fold_dir)

    all_test_scores = np.stack(all_test_scores, axis=0)
    mean, std = np.mean(all_test_scores, axis=0), np.std(all_test_scores, axis=0)
    print(f'{args.num_folds} fold average: {mean} +/- {std}')
