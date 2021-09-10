import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from models import proto_loss, exemplar_loss


def model_evaluate(model, val_loader, device, loss_function, eval_epochs):
    model.eval()
    eval_losses = []
    eval_accs = []
    for epoch in range(eval_epochs):
        for episode, (frame_idx_tensor, query_nouns_idx_tensor, support_nouns_idx_tensor,
                      support_nouns_total_counts_tensor) in tqdm(enumerate(val_loader), position=0, leave=True):

            query_nouns_idx_tensor, support_nouns_idx_tensor, support_nouns_total_counts_tensor \
                = query_nouns_idx_tensor.to(device), support_nouns_idx_tensor.to(device), \
                  support_nouns_total_counts_tensor.to(device)
            Hq_fused = model(query_nouns_idx_tensor)
            Hs_fused = model(support_nouns_idx_tensor)
            if loss_function == 'proto_loss':
                output = proto_loss(Hq_fused, Hs_fused)
            else:
                output = exemplar_loss(Hq_fused, Hs_fused)
            eval_losses.append(output['loss'])
            eval_accs.append(output['acc'])
    acc = torch.tensor(eval_accs).mean()
    loss = torch.tensor(eval_losses).mean()
    return {
        'acc': acc,
        'loss': loss
    }


def model_test(model, query_noun_loader, frame_loader, device, co_occurrence_mat,
               loss_function, topk=20):
    model.eval()

    Hss = []
    Hqs = []
    ns_counts = []
    accs_noun = []
    accs_frame = []

    noun_prediction_df = {
        'frame': [],
        'ground truth novel nouns': [],
        'predicted novel nouns': []
    }
    frame_prediction_df = {
        'query noun': [],
        'ground truth extended frames': [],
        'predicted extended frames': []
    }
    query_nouns = list(query_noun_loader.dataset.learning_df['query noun'])
    frames = list(frame_loader.dataset.learning_df['frame'])

    for _, query_nouns_idx_tensor in tqdm(enumerate(query_noun_loader), position=0, leave=True):
        with torch.no_grad():
            query_nouns_idx_tensor = query_nouns_idx_tensor.to(device)
            Hq_fused = model(query_nouns_idx_tensor)
            Hqs.append(Hq_fused.detach().cpu())

    for i, (support_nouns_idx_tensor, support_nouns_total_counts_tensor, novel_nouns_idx) \
            in tqdm(enumerate(frame_loader), position=0, leave=True):
        with torch.no_grad():
            support_nouns_idx_tensor = support_nouns_idx_tensor.to(device)
            Hs_fused = model(support_nouns_idx_tensor)
            Hss.append(Hs_fused.detach().cpu())
            ns_counts.append(support_nouns_total_counts_tensor)

    Hqs = torch.cat(Hqs, dim=0)
    Hss = torch.cat(Hss, dim=0)
    ns_counts = torch.cat(ns_counts, dim=0)
    N_nouns = Hqs.shape[0]
    N_frames = Hss.shape[0]
    joint_hidden_dim = Hqs.shape[-1]
    n_s = Hss.shape[1]

    if loss_function == 'proto_loss':
        Cs = Hss.mean(dim=1)
        euc_dists = torch.pow(torch.cdist(Hqs, Cs), 2)  # shape (N_nouns, N_frames)
        ns_counts = ns_counts.view(1, -1).repeat(N_nouns, 1).float()  # shape (N_nouns, N_frames)
        log_p_f = F.log_softmax(-euc_dists, dim=-1)  # shape (N_nouns, N_frames)
        # log_p_n = F.log_softmax(-torch.mul(euc_dists, ns_counts), dim=0)
        log_p_n = F.log_softmax(-euc_dists, dim=0)
        predicted_frames_idx = torch.argsort(-log_p_f, dim=-1)  # shape (N_nouns, N_frames)
        predicted_nouns_idx = torch.transpose(torch.argsort(-log_p_n, dim=0), 0, 1)  # shape (N_frames, N_nouns)
    else:
        euc_dists = [torch.pow(torch.cdist(Hqs,
                                           Hss[i, :, :].view(n_s, joint_hidden_dim)).view(N_nouns, n_s), 2) for i in
                     range(N_frames)]
        euc_dists = torch.cat(euc_dists, dim=-1)  # shape (N_nouns, N_frames*n_s)

        # ns_counts = ns_counts.view(1, -1, 1).repeat(N_nouns, 1, n_s).view(N_nouns, N_frames * n_s)
        log_p_f = torch.sum(F.log_softmax(-euc_dists, dim=-1).view(N_nouns, N_frames, n_s),
                            dim=-1)  # shape (N_nouns, N_frames)
        log_p_n = torch.sum(F.log_softmax(-euc_dists, dim=0).view(N_nouns, N_frames, n_s),
                            dim=-1)
        log_p_n = torch.transpose(log_p_n, 0, 1)    # shape (N_frames, N_nouns)
        predicted_frames_idx = torch.argsort(-log_p_f, dim=-1)  # shape (N_nouns, N_frames)
        predicted_nouns_idx = torch.argsort(-log_p_n, dim=-1)  # shape (N_frames, N_nouns)

    ground_truth_frame_idx = []
    ground_truth_noun_idx = []
#     import pdb; pdb.set_trace()
    for i in range(predicted_frames_idx.shape[0]):
        ground_truth_frame_idx.append((co_occurrence_mat[i, :] == 1).nonzero(as_tuple=True)[0])
    for i in range(predicted_nouns_idx.shape[0]):
        ground_truth_noun_idx.append((co_occurrence_mat[:, i] == 1).nonzero(as_tuple=True)[0])
    precisions_f, auc_f = roc(predicted_frames_idx, ground_truth_frame_idx)
    precisions_n, auc_n = roc(predicted_nouns_idx, ground_truth_noun_idx)

    for i in range(predicted_frames_idx.shape[0]):
        query_noun = query_nouns[i]
        row_ground_truth_frames = [frames[k.item()] for k in ground_truth_frame_idx[i]]
        row_predicted_frames_idx = predicted_frames_idx[i][:max(topk, len(row_ground_truth_frames))]
        row_predicted_frames = [frames[k.item()] for k in row_predicted_frames_idx]
        row_acc_frame = float(
            len(np.intersect1d(ground_truth_frame_idx[i],
                               row_predicted_frames_idx))) / len(row_ground_truth_frames)
        accs_frame.append(row_acc_frame)
        frame_prediction_df['query noun'].append(query_noun)
        frame_prediction_df['ground truth extended frames'].append(row_ground_truth_frames)
        frame_prediction_df['predicted extended frames'].append(row_predicted_frames)
    for i in range(predicted_nouns_idx.shape[0]):
        frame = frames[i]
        row_ground_truth_nouns = [query_nouns[k.item()] for k in ground_truth_noun_idx[i]]
        row_predicted_nouns_idx = predicted_nouns_idx[i][:max(topk, len(row_ground_truth_nouns))]
        row_predicted_nouns = [query_nouns[k.item()] for k in row_predicted_nouns_idx]
        row_acc_noun = float(
            len(np.intersect1d(ground_truth_noun_idx[i],
                               row_predicted_nouns_idx))) / len(row_ground_truth_nouns)
        accs_noun.append(row_acc_noun)
        noun_prediction_df['frame'].append(frame)
        noun_prediction_df['ground truth novel nouns'].append(row_ground_truth_nouns)
        noun_prediction_df['predicted novel nouns'].append(row_predicted_nouns)

    acc_noun = np.array(accs_noun).mean()
    acc_frame = np.array(accs_frame).mean()
    return {
            'acc_noun': acc_noun,
            'acc_frame': acc_frame,
            'precisions_noun': precisions_n,
            'precisions_frame': precisions_f,
            'auc_noun': auc_n,
            'auc_frame': auc_f,
            'noun_prediction_df': noun_prediction_df,
            'frame_prediction_df': frame_prediction_df,
            'Hqs': Hqs,
            'Hss': Hss
            }


def roc(predictions_mat, ground_truths, k_step=10):
    # sorted_col_idx: shape (N_f, N_n) or (N_n, N_f)
    n_classes = predictions_mat.shape[-1]
    precisions = []
    for k in range(1, 1+n_classes, k_step):
        k_precisions = []
        for i in range(predictions_mat.shape[0]):
            predicted_cols = predictions_mat[i][:k]
            k_precision = float(
                len(np.intersect1d(predicted_cols,
                                   ground_truths[i]))) / len(ground_truths[i])
            k_precisions.append(k_precision)
        k_precision = np.array(k_precisions)
        precisions.append(k_precision)
    precisions = np.array(precisions)
    auc = precisions.mean()
    return np.array(precisions), auc
