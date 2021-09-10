import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LikelihoodNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoders = {}
        self.num_modes = len(config.modalities)
        self.hidden_dim_0 = 0
        if 'vis' in config.modalities:
            encoders['vis'] = VisualEncoder(config)
            self.hidden_dim_0 += config.vis_hidden_dim
        if 'ont' in config.modalities:
            encoders['ont'] = OntologicalEncoder(config)
            self.hidden_dim_0 += config.ont_hidden_dim
        if 'ling' in config.modalities:
            encoders['ling'] = LinguisticsEncoder(config)
            self.hidden_dim_0 += config.ling_hidden_dim
        self.encoders = nn.ModuleDict(encoders)
        self.hidden_dim_1 = config.hidden_dim_1
        self.hidden_dim_2 = config.hidden_dim_2
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden_dim_0, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        )

    def forward(self, noun_idx_tensor):
        Hs = []
        for mode, encoder in self.encoders.items():
            H = encoder(noun_idx_tensor)
            Hs.append(H)
        Hs = torch.cat(Hs, dim=-1)  # shape (B, n, n_mode * mode_hidden_dim)
        Hs_fused = self.fuse(Hs.float())  # shape (B, n, joint_hidden_dim)

        return Hs_fused


def proto_loss(Hq_fused, Hs_fused):
    B = Hs_fused.shape[0]
    n_q = Hq_fused.shape[1]
    P_s = torch.mean(Hs_fused, dim=1)  # shape (B, h_dim)
    # compute the tensor of exponential distances between each pair of (query, support) nouns
    qs_dists_tensor = euclidean_dist(Hq_fused.view(B * n_q, -1),
                                     P_s).view(B, n_q, B)  # shape (B, n_q, B)
    log_p_f = F.log_softmax(-qs_dists_tensor, dim=-1)  # shape (B, n_q, B)
    log_p_n = F.log_softmax(-qs_dists_tensor, dim=0)  # shape (B, n_q, B)
    losses = []
    n_correct = 0
    for i in range(B):
        losses.append(torch.sum(log_p_f[i, :, i]))
        losses.append(torch.sum(log_p_n[i, :, i]))
        for j in range(n_q):
            if torch.argmax(log_p_f[i, j, :]) == i:
                n_correct += 1
    acc = float(n_correct) / (B * n_q)
    loss = -torch.stack(losses, dim=0).sum()
    return {'loss': loss,
            'joint prob': qs_dists_tensor,
            'acc': acc
            }


def exemplar_loss(Hq_fused, Hs_fused):
    B = Hs_fused.shape[0]
    n_q = Hq_fused.shape[1]
    n_s = Hs_fused.shape[1]

    # compute the tensor of exponential distances
    #     import pdb; pdb.set_trace()
    qs_dists_tensor = [euclidean_dist(Hq_fused.view(B * n_q, -1), Hs_fused[i, :, :].view(n_s, -1)) \
                           .view(B, n_q, n_s) for i in range(B)]  # shape (B, n_q, B*n_s)
    qs_dists_tensor = torch.cat(qs_dists_tensor, dim=-1)    # shape (B, n_q, B, n_s)

    log_p_f = F.log_softmax(-qs_dists_tensor, dim=-1).view(B, n_q, B, n_s).sum(-1)  # shape (B, n_q, B)
    log_p_n = F.log_softmax(-qs_dists_tensor, dim=0).view(B, n_q, B, n_s).sum(-1)  # shape (B, n_q, B)
    losses = []
    n_correct = 0
    for i in range(B):
        losses.append(torch.sum(log_p_f[i, :, i]))
        losses.append(torch.sum(log_p_n[i, :, i]))
        for j in range(n_q):
            if torch.argmax(log_p_f[i, j, :]) == i:
                n_correct += 1
    acc = float(n_correct) / (B * n_q)
    loss = -torch.stack(losses, dim=0).sum()
    return {'loss': loss,
            'joint prob': qs_dists_tensor,
            'acc': acc
            }


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vis_embeddings = nn.Embedding.from_pretrained(config.imagenet_embeddings, max_norm=3.0, freeze=True)

    def forward(self, nouns_idx_tensor):
        Hs = self.vis_embeddings(nouns_idx_tensor)
        return Hs


class OntologicalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conceptnet_embeddings = nn.Embedding.from_pretrained(config.conceptnet_embeddings, max_norm=3.0, freeze=True)

    def forward(self, nouns_idx_tensor):
        Hs = self.conceptnet_embeddings(nouns_idx_tensor)
        return Hs


class LinguisticsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.histwords_embeddings = nn.Embedding.from_pretrained(config.histwords_embeddings, freeze=True)

    def forward(self, nouns_idx_tensor):
        Hs = self.histwords_embeddings(nouns_idx_tensor)
        return Hs


# def roc(predictions_mat, ground_truths):
#     # sorted_col_idx: shape (N_f, N_n) or (N_n, N_f)
#     n_classes = predictions_mat.shape[-1]
# #     import pdb; pdb.set_trace()
#     precisions = []
#     for k in range(1, 1+n_classes):
#         k_precisions = []
#         for i in range(predictions_mat.shape[0]):
#             predicted_cols = predictions_mat[i][:k]
#             precision = float(
#                 len(np.intersect1d(predicted_cols,
#                                    ground_truths[i]))) / k
#             k_precisions.append(precision)
#         k_precision = np.array(k_precisions).mean()
#         precisions.append(k_precision)
#     precisions = np.array(precisions)
#     auc = precisions.mean()
#     return np.array(precisions), auc
