import os
import esm
import torch
import warnings
import argparse
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedKFold
warnings.simplefilter('ignore')
class PDB(Dataset):
    def __init__(self, mode='train', fold=-1, root='./data/Data_245', self_cycle=False):
        self.root = root
        assert mode in ['train', 'val', 'test']
        if mode in ['train', 'val']:
            with open(f'{self.root}/train.pkl', 'rb') as f:
                self.samples = pk.load(f)
        else:
            with open(f'{self.root}/test.pkl', 'rb') as f:
                self.samples = pk.load(f)
        self.data = []
        idx = np.load(f'{self.root}/cross-validation.npy')
        cv = 10
        inter = len(idx) // cv
        ex = len(idx) % cv
        if mode == 'train':
            order = []
            for i in range(cv):
                if i == fold:
                    continue
                order += list(idx[i * inter:(i + 1) * inter + ex * (i == cv - 1)])
        elif mode == 'val':
            order = list(idx[fold * inter:(fold + 1) * inter + ex * (fold == cv - 1)])
        else:
            order = list(range(len(self.samples)))
        order.sort()
        tbar = tqdm(order)
        for i in tbar:
            tbar.set_postfix(chain=f'{self.samples[i].name}')
            self.samples[i].load_feat(self.root)
            self.samples[i].load_dssp(self.root)
            self.samples[i].load_adj(self.root, self_cycle)
            self.data.append(self.samples[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        min_len = min(seq.feat.size(0), seq.dssp.size(0), seq.esm_if.size(0))
        feat = seq.feat[:min_len]
        dssp = seq.dssp[:min_len]
        esm_if = seq.esm_if[:min_len]

        feat = torch.cat([feat, dssp,esm_if], 1)
        edge_index = seq.adj.nonzero().t().contiguous()
        edge_attr = seq.edge[edge_index[0], edge_index[1]]
        return {
            'feat': feat,
            'label': seq.label,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/Data_245', help='dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    args = parser.parse_args()
    root = args.root
    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
    
    os.system(f'cd {root} && mkdir PDB purePDB feat dssp graph fasta esm_if')

    train_=r'D:\MsgaBpred\data\Data_245\train_labeled_fasta'
    test_=r'D:\MsgaBpred\data\Data_245\test_labeled_fasta'
    trainset, valset, testset = [], [], []
    process_train_fasta_directory(train_,root,None,device)

    with open(f'{root}/train.pkl', 'rb') as f:
        train_dataset = pk.load(f)
    for i in train_dataset:
        trainset.append(i)

    filt_data = []
    for i in train_dataset:
        if  i.label.sum() > 0:
            filt_data.append(i)

    idx = np.random.permutation(len(filt_data))
    np.save(f'{root}/cross-validation.npy', idx)


    process_test_fasta_directory(test_,root,None,device)
    with open(f'{root}/test.pkl', 'rb') as f:
        test_dataset = pk.load(f)
    for i in test_dataset:
        testset.append(i)

