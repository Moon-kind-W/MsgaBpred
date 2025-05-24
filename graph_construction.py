import torch
import numpy as np


ID = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}


def calcPROgraph(seq, coord, dseq=3, dr=10, dlong=5, k=10):

    nodes = coord.shape[0]
    adj = torch.zeros((nodes, nodes))
    edge_dim = 21 * 2 + 2 * dseq + 3
    E = torch.zeros((nodes, nodes, edge_dim))

    dist = torch.cdist(coord, coord, p=2)

    knn = dist.argsort(dim=1)[:, 1:k + 1]

    for i in range(nodes):
        for j in range(nodes):
            not_edge = True
            dij_seq = abs(i - j)
            if dij_seq < dseq:
                E[i][j][41 + i - j + dseq] = 1
                not_edge = False

            if dist[i][j] < dr and dij_seq >= dlong:
                E[i][j][41 + 2 * dseq] = 1
                not_edge = False

            if j in knn[i] and dij_seq >= dlong:
                E[i][j][42 + 2 * dseq] = 1
                not_edge = False

            if not_edge:
                continue

            adj[i][j] = 1

            E[i][j][ID.get(seq[i], 20)] = 1
            E[i][j][21 + ID.get(seq[j], 20)] = 1

            E[i][j][43 + 2 * dseq] = dij_seq
            E[i][j][44 + 2 * dseq] = dist[i][j]

    adj = adj.to_sparse()
    E = E.to_sparse()
    return {'adj': adj, 'edge': E}
