import os
import torch
import random
import warnings
import argparse
import numpy as np
import pickle as pk
import pytorch_lightning as pl
from tool import METRICS
from model import GraphBepi
from dataset import PDB, collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import pandas as pd
import pickle
warnings.simplefilter('ignore')

def seed_everything(seed=43):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def aggregate_results(args):
    log_name = f'{args.dataset}_{args.tag}'
    result_path = f"./model/{log_name}"
    all_results = []
    for fold in range(10):
        result_file = os.path.join(result_path, f"result_fold_{fold}.pkl")
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pk.load(f)
                all_results.append(result)

    average_result = {}
    for key in all_results[0][0].keys():
        values = [result[0][key] for result in all_results]
        average_result[key] = np.mean(values)

    final_result_file = os.path.join(result_path, "final_average_result.pkl")
    with open(final_result_file, 'wb') as f:
        pk.dump(average_result, f)
    print("Final average test results:")
    print(average_result)
def find_best_fold(args):
    log_name = f'{args.dataset}_{args.tag}'
    result_path = f"./model/{log_name}"

    best_fold = -1
    best_val_auroc = -1
    best_threshold = 0.5

    for fold in range(10):
        checkpoint_file = os.path.join(result_path, f"model_{fold}.ckpt")
        result_file = os.path.join(result_path, f"result_fold_{fold}.pkl")

        if os.path.exists(checkpoint_file) and os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pk.load(f)
                test_auroc = result[0]['test_AUROC']
                test_threshold = result[0].get('test_threshold', 0.5)

                if test_auroc > best_val_auroc:
                    best_val_auroc = test_auroc
                    best_fold = fold
                    best_threshold = test_threshold
    return best_fold, best_threshold
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument('--seed', type=int, default=43, help='random seed.')
    parser.add_argument('--batch', type=int, default=8, help='batch size.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dim.')
    parser.add_argument('--epochs', type=int, default=50, help='max number of epochs.')
    parser.add_argument('--dataset', type=str, default='Data_245', help='dataset name.')
    parser.add_argument('--logger', type=str, default='./log', help='logger path.')
    parser.add_argument('--tag', type=str, default='MsgaBpred', help='logger name.')
    parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint to resume training from.')
    parser.add_argument('--dropout', type=float, default=0.30, help='')
    parser.add_argument('--output', type=str, default='./results', help='output directory for saving predictions.')
    args = parser.parse_args()
    device = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    seed_everything(args.seed)
    root = f'./data/{args.dataset}'

    all_fold_results = []
    for fold in range(10):
        print(f"Training and evaluating fold {fold}...")

        trainset = PDB(mode='train', fold=fold, root=root)
        valset = PDB(mode='val', fold=fold, root=root)
        testset = PDB(mode='test', fold=fold, root=root)

        train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(valset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        log_name = f'{args.dataset}_{args.tag}'
        metrics = METRICS(device)

        model = MsgabBpred(
            esmc_dim=2560,
            hidden_dim=args.hidden,
            exfeat_dim=13,
            esmif_dim=512,
            edge_dim=51,
            augment_eps=0.1,
            dropout=args.dropout,
            lr=args.lr,
            metrics=metrics,
            result_path=f'./model/{log_name}',
        )

        mc = ModelCheckpoint(
            dirpath=f'./model/{log_name}/',
            filename=f'model_{fold}',
            monitor='val_AUROC',
            mode='max',
            save_weights_only=False,
            save_top_k=1,
            every_n_epochs=1,
        )

        cb = [mc]

        trainer = pl.Trainer(
            gpus=[args.gpu] if args.gpu != -1 else None,
            max_epochs=args.epochs,
            callbacks=cb,
            logger=None,
            check_val_every_n_epoch=1,
            resume_from_checkpoint=args.resume,
        )

        if os.path.exists(f'./model/{log_name}/model_{fold}.ckpt'):
            os.remove(f'./model/{log_name}/model_{fold}.ckpt')

        trainer.fit(model, train_loader, val_loader)
        trainer = pl.Trainer(gpus=[args.gpu], logger=None)
        result = trainer.test(model, test_loader)
        result_file = f'./model/{log_name}/result_fold_{fold}.pkl'
        with open(result_file, 'wb') as f:
            pk.dump(result, f)
        all_fold_results.append(result)
    aggregate_results(args)
    best_fold, best_threshold = find_best_fold(args)
    print(f"Best fold: {best_fold}, Best val_auroc model's threshold: {best_threshold}")

    log_name = f'{args.dataset}_{args.tag}'
    best_model_path = f'./model/{log_name}/model_{best_fold}.ckpt'
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    testset = PDB(mode='test', fold=best_fold, root=root)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    trainer = pl.Trainer(gpus=[args.gpu], logger=None)
    test_results = trainer.test(model, test_loader)
    os.makedirs(args.output, exist_ok=True)
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
        labeli = torch.where(predi > best_threshold, 1, 0).bool()
        label = y[idx]
        df = pd.DataFrame({'resn': list(seqi), 'score': predi, 'Label':label,'is epitope': labeli})
        df.to_csv(f'{args.output}/{testset.data[i].name}.csv', index=False)


