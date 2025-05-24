import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import AE
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence

from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.utils import to_dense_batch
import dataset
from torch.nn import Parameter
import numpy as np



class Additive_AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(Additive_AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(x * attn_weights, dim=1)  # (batch_size, hidden_dim)
        return weighted_output, attn_weights.squeeze(-1)




class MutiscaleGCN(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            edge_dim,
            num_layers=1,
            dropout=0.3,
    ):
        super(MutiscaleGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 多尺度 GCN 配置
        self.kernel_sizes = [1,2,3]
        self.num_scales = len(self.kernel_sizes)

        # 特征融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))


        self.projection = nn.Linear(in_channels, hidden_channels)


        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):

            layer = nn.ModuleDict()
            for k in self.kernel_sizes:
                layer[f'conv_{k}'] = GCNConv(
                    in_channels if _ == 0 else hidden_channels,
                    hidden_channels
                )
            self.gcn_layers.append(layer)
            self.gcn_layers.append(nn.Linear(hidden_channels * self.num_scales, hidden_channels))

    def propagate_multiscale(self, x, edge_index, layer_dict):
        scale_features = []
        num_nodes = x.size(0)
        device = x.device
        for k, conv in layer_dict.items():
            hops = int(k.split('_')[-1])
            if hops == 1:
                x_scale = conv(x, edge_index)
            else:
                adj = torch.sparse_coo_tensor(
                    edge_index,
                    torch.ones(edge_index.size(1), device=device),
                    size=(num_nodes, num_nodes)
                )

                current_adj = adj.clone()
                for _ in range(hops - 1):
                    current_adj = torch.sparse.mm(current_adj, adj)
                current_edge_index = current_adj.coalesce().indices()
                x_scale = conv(x, current_edge_index)

            x_scale = F.relu(x_scale)
            x_scale = F.dropout(x_scale, p=self.dropout, training=self.training)
            scale_features.append(x_scale)
        return scale_features

    def forward(self, x, edge_index, edge_attr=None):
        if edge_index.size(1) == 0:
            print("Empty edge_index detected!")
        original_x = x
        original_x = self.projection(original_x)
        for i in range(0, self.num_layers * 2, 2):

            layer_dict = self.gcn_layers[i]
            scale_features = []
            scale_features = self.propagate_multiscale(x, edge_index, layer_dict)
            x = torch.cat(scale_features, dim=-1)
            if i + 1 < len(self.gcn_layers):
                x = self.gcn_layers[i + 1](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        fused_x = self.fusion_weight * x + (1 - self.fusion_weight) * original_x

        return fused_x

class MsgaBpred(pl.LightningModule):
    def __init__(
        self,
        esmc_dim=2560, hidden_dim=128,
        exfeat_dim=13,
        esmif_dim=512,
        edge_dim=51,
        augment_eps=0.1, dropout=0.5,
        lr=1e-5, metrics=None, result_path=None
    ):
        super().__init__()
        self.metrics = metrics
        self.path = result_path
        self.loss_fn = nn.BCELoss()
        self.esmc_dim =esmc_dim
        self.exfeat_dim = exfeat_dim
        self.esmif_dim = esmif_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias = False
        self.W_v1 = nn.Sequential(
            nn.Linear(esmc_dim, hidden_dim * 5,bias=bias),
            nn.LayerNorm(hidden_dim*5),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 5, hidden_dim*2,bias=bias),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2,hidden_dim,bias=bias),
            nn.LayerNorm(hidden_dim)
        )
        self.W_v2 = nn.Sequential(
            nn.Linear(esmif_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.W_u1 = AE(exfeat_dim, hidden_dim, hidden_dim, bias=bias)

        self.edge_linear = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 4, bias=True),
            nn.ELU(),
        )
        self.mutiscalegcn= MutiscaleGCN(
            in_channels= 3*hidden_dim,
            hidden_channels=hidden_dim,
            edge_dim=hidden_dim // 4,
            num_layers=2,
            dropout=dropout,
        )
        self.Attention = Additive_AttentionLayer(hidden_dim*3)
        self.mlp = nn.Sequential(
            nn.Linear( hidden_dim*4,  hidden_dim*2, bias=True),  # 输入维度为 3 * hidden_dim
            nn.ReLU(),
            nn.Linear( hidden_dim*2, 1, bias=True),
            nn.Sigmoid()
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, V, edge_index, edge_attr):
        for i, e in enumerate(edge_index):
            num_nodes = V[i].shape[0]
            if e.max() >= num_nodes or e.min() < 0:
                print(
                    f"Error: edge_index[{i}] has invalid indices. Max: {e.max()}, Min: {e.min()}, Num nodes: {num_nodes}")
                raise ValueError("edge_index contains out-of-bounds indices.")
        h = []
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask = V.sum(-1) != 0
        if self.training and self.augment_eps > 0:
            aug = torch.randn_like(V)
            aug[~mask] = 0
            V = V + self.augment_eps * aug
        mask = mask.sum(1)
        esmc_feats = self.W_v1(V[:, :, :self.esmc_dim])
        dssp_feats = self.W_u1(V[:, :, self.esmc_dim:self.esmc_dim + self.exfeat_dim])
        esmif_feats = self.W_v2(
            V[:, :, self.esmc_dim + self.exfeat_dim:self.esmc_dim + self.exfeat_dim + self.esmif_dim])
        feats = torch.cat([esmc_feats, dssp_feats, esmif_feats], dim=-1)
        x_gcns = []
        for i in range(len(V)):
            E = edge_attr[i].unsqueeze(0)
            E = E.permute(1, 0, 2)
            E = E.squeeze(1)
            E = self.edge_linear(E)
            x_gcn = feats[i, :mask[i]]
            x_gcn = self.mutiscalegcn(x_gcn, edge_index[i], E)
            x_gcns.append(x_gcn)
        x_gcns = pad_sequence(x_gcns, batch_first=True)
        attn_feats = [feats[i, :mask[i]] for i in range(len(feats))]
        attn_feats = pad_sequence(attn_feats,batch_first=True)
        attention_output, _ = self.Attention(attn_feats)
        attention_output = attention_output.unsqueeze(1).expand(-1, attn_feats.size(1), -1)
        h = torch.cat([x_gcns,attention_output],dim=-1)
        return self.mlp(h)





    def training_step(self, batch, batch_idx):
        feat, edge_index, edge_attr, y = batch
        pred = self(feat, edge_index, edge_attr).squeeze(-1)
        new_y = []
        start_idx = 0
        for feat_ in feat:
            end_idx_ = feat_.shape[0]
            end_idx = start_idx + end_idx_
            new_y.append(y[start_idx:end_idx])
            start_idx += end_idx_
        y = pad_sequence(new_y, batch_first=True)
        loss = self.loss_fn(pred, y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result = self.metrics.calc_prc(pred.detach().clone(), y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        feat, edge_index, edge_attr, y = batch
        pred = self(feat, edge_index, edge_attr).squeeze(-1)
        return pred, y
    def validation_epoch_end(self,outputs):
        pred,y, contrastive_losses=[],[],[]
        for i,j in outputs:
            y.append(j)
            pred.append(i)
        pred = torch.cat(pred, 1).view(-1)
        y=torch.cat(y,0)
        loss = self.loss_fn(pred, y.float())
        self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result=self.metrics(pred.detach().clone(),y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        feat, edge_index, edge_attr, y = batch
        pred = self(feat, edge_index, edge_attr).squeeze(-1)
        return pred, y

    def test_epoch_end(self, outputs):
        pred, y = [], []
        for i, j in outputs:
            pred.append(i)
            y.append(j)
        pred = torch.cat(pred, 1).view(-1)
        y = torch.cat(y, 0)
        loss = self.loss_fn(pred, y.float())
        if self.path:
            if not os.path.exists(self.path):
                os.system(f'mkdir -p {self.path}')
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('test_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_recall', result['RECALL'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)








