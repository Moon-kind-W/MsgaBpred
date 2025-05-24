import os
import esm
import time
import torch
import random
import warnings
import argparse
import shutil
import numpy as np
import pandas as pd
import pickle as pk
import pytorch_lightning as pl
from tqdm import tqdm
from tool import METRICS
from model import GraphBepi
from dataset import PDB, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from utils import process_test_fasta_directory


warnings.simplefilter('ignore')


def seed_everything(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    seed_everything(args.seed)

    # Prepare directories
    tmp_root = './data/tmp'
    if not os.path.exists(args.output):
        os.makedirs(args.output)



    os.makedirs(f'{tmp_root}/purePDB', exist_ok=True)
    os.makedirs(f'{tmp_root}/feat', exist_ok=True)
    os.makedirs(f'{tmp_root}/dssp', exist_ok=True)
    os.makedirs(f'{tmp_root}/graph', exist_ok=True)
    os.makedirs(f'{tmp_root}/fasta', exist_ok=True)
    os.makedirs(f'{tmp_root}/esm_if', exist_ok=True)



    # Process input file



    print('Processing PDB file...')
    test_root = './data/tmp/test_labeled_fasta/'
    out_dir = './data/tmp'
    process_test_fasta_directory(test_root, out_dir, None, device)
    # Prepare test dataset
    with open(f'{tmp_root}/test.pkl', 'rb') as f:
        chains = pk.load(f)

    idx = np.array(range(len(chains)))
    np.save(f'{tmp_root}/cross-validation.npy', idx)

    print('Loading test data...')
    testset = PDB(mode='test', root=tmp_root)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize model with same parameters as train.py
    metrics = METRICS(device)
    model = MsgaBpred(
        esmc_dim=2560,
        hidden_dim=args.hidden,
        exfeat_dim=13,
        esmif_dim=512,
        edge_dim=51,
        augment_eps=0.1,
        dropout=args.dropout,
        lr=args.lr,
        metrics=metrics,
        result_path=args.output,
    )

    # Load pretrained weights
    checkpoint_path = f'./model/{args.dataset}_{args.tag}/model_{args.fold}.ckpt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])

    # Initialize logger (optional)
    if args.log:
        wandb.init(project="GraphBepi", name=f"{args.dataset}_{args.tag}_test", config=args)
        logger = WandbLogger(project="GraphBepi", name=f"{args.dataset}_{args.tag}_test")

    print('Predicting...')
    trainer = pl.Trainer(gpus=[args.gpu] if args.gpu != -1 else None, logger=logger if args.log else None)
    result = trainer.test(model, test_loader)
    test_results = torch.load(f'{model.path}/result.pkl')
    pred = test_results['pred']
    y = test_results['gt']
    IDX = []
    for i in range(len(testset)):
        IDX += [i] * len(testset.data[i])
    IDX = torch.LongTensor(IDX)
    for i in range(len(testset)):
        idx = IDX == i
        predi = pred[idx]
        seqi = testset.data[i].sequence
        labeli = torch.where(predi > args.threshold, 1, 0).bool()
        label = y[idx]
        df = pd.DataFrame({'resn': list(seqi), 'score': predi, 'Label': label, 'is epitope': labeli})
        df.to_csv(f'{args.output}/{testset.data[i].name}.csv', index=False)

    if os.path.exists(f'{args.output}/result.pkl'):
        os.remove(f'{args.output}/result.pkl')

    print('Prediction completed. Results saved to:', args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device')
    parser.add_argument('--seed', type=int, default=43, help='random seed')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dim')
    parser.add_argument('--dataset', type=str, default='Data_245', help='dataset name')
    parser.add_argument('--tag', type=str, default='MsgaBpred', help='model tag')
    parser.add_argument('--fold', type=int, default='6', help='best fold')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--threshold', type=float, default=0.102, help='prediction threshold')
    parser.add_argument('--output', type=str, default='./output', help='output path')
    parser.add_argument('--log', action='store_true', help='enable wandb logging')

    args = parser.parse_args()
    main(args)