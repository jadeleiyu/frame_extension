import logging
import os
import pickle
from collections import namedtuple
from copy import deepcopy
from datetime import date
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from dataset import GSNFrameDataset
from models import LikelihoodNetwork, proto_loss, exemplar_loss
from evaluation import model_evaluate, model_test

torch.manual_seed(2)

Config = namedtuple('parameters',
                    ['noun2idx',
                     'start_decade', 'end_decade',
                     'imagenet_embeddings',
                     'conceptnet_embeddings',
                     'histwords_embeddings',
                     'loss_function',
                     'hidden_dim_1', 'hidden_dim_2',
                     'vis_hidden_dim', 'ont_hidden_dim', 'ling_hidden_dim',
                     'modalities']
                    )


def main():
    start_decade = 1850
    end_decade = 1990
    train_batch_size = 64
    val_batch_size = 64
    test_batch_size = 64
    num_workers = 1
    n_s = 8
    n_q = 2
    max_epochs_train = 100
    max_epochs_eval = 10
    eval_every = 5
    vis_hidden_dim = 1000
    ont_hidden_dim = 300
    ling_hidden_dim = 300
    hidden_dim_1 = 400
    hidden_dim_2 = 200
    loss_function = 'proto_loss'
    learning_rate = 1e-5
    weight_decay = 0
    scheduler_step_size = 500
    scheduler_gamma = 0.9
    iters_to_accumulate = 2
    use_pretrained_model = False
    cuda = True
    log_dir = '/h/19/jadeleiyu/frame_extension/log/'
    model_dir = '/h/19/jadeleiyu/frame_extension/models/'
    prediction_dir = '/h/19/jadeleiyu/frame_extension/predictions/'
    modalities = ['vis', 'ont', 'ling']

    noun2idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/support_noun2idx.p',
                                'rb'))  # a common noun2idx lookup dict shared across all modalities
    list_converter = {'novel nouns': eval, 'most similar novel nouns': eval, 'least similar novel nouns': eval,
                      'established nouns': eval, 'ground truth extended frames': eval,
                      'support nouns': eval, 'ground truth novel nouns': eval, 'support noun counts': eval,
                      'established noun counts': eval}

    cuda_available = torch.cuda.is_available()
    if cuda and cuda_available:
        device = torch.device("cuda")
        print("using gpu acceleration")
    else:
        device = torch.device("cpu")
        print("using cpu")

    print('performing learning by decade...')
    for decade in range(start_decade, end_decade + 10, 10):
        print('begin learning in decade {}s'.format(decade))
        print('preparing learning data...')

        decade_training_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/training_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_training_df = decade_training_df.reset_index()
        decade_eval_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/evaluation_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_noun_test_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/noun_eval_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_frame_test_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/frame_eval_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_co_occurrence_mat = torch.load('/h/19/jadeleiyu/frame_extension/data/gsn/co_occurrence_mat_{}s.pt'
                                              .format(decade))

        train_ds = GSNFrameDataset(decade_training_df, noun2idx, n_q, n_s, learning_type='train')
        val_ds = GSNFrameDataset(decade_eval_df, noun2idx, n_q, n_s, learning_type='val')
        noun_test_ds = GSNFrameDataset(decade_noun_test_df, noun2idx, n_q, n_s, learning_type='test_noun')
        frame_test_ds = GSNFrameDataset(decade_frame_test_df, noun2idx, n_q, n_s, learning_type='test_frame')

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=train_batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_ds, shuffle=True, batch_size=val_batch_size, num_workers=num_workers)
        noun_test_loader = DataLoader(noun_test_ds, shuffle=False, batch_size=test_batch_size, num_workers=num_workers)
        frame_test_loader = DataLoader(frame_test_ds, shuffle=False, batch_size=test_batch_size,
                                       num_workers=num_workers)

        conceptnet_embeddings = torch.tensor(pickle.load(
            open('/h/19/jadeleiyu/frame_extension/data/cnp/cnp_hist_embeddings_{}'.format(decade), 'rb')))
        histwords_embeddings = torch.tensor(np.load(
            '/h/19/jadeleiyu/frame_extension/data/histwords_embeddings/embeddings_{}.npy'.format(decade)))
        imagenet_embeddings = torch.load('/h/19/jadeleiyu/frame_extension/data/img/noun_image_features.pt')

        print('Generating configuration...')
        config = Config(
            noun2idx=noun2idx,
            start_decade=start_decade,
            end_decade=end_decade,
            conceptnet_embeddings=conceptnet_embeddings,
            histwords_embeddings=histwords_embeddings,
            imagenet_embeddings=imagenet_embeddings,
            loss_function=loss_function,
            vis_hidden_dim=vis_hidden_dim,
            ont_hidden_dim=ont_hidden_dim,
            ling_hidden_dim=ling_hidden_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            modalities=modalities,
        )

        print('initializing model...')
        model = LikelihoodNetwork(config).to(device)
        if use_pretrained_model:
            model_fn = '{}_{}_{}.pt'.format(decade, loss_function, '-'.join(modalities))
            model.load_state_dict(torch.load(model_dir + model_fn))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        scaler = GradScaler()

        print('begin training...')
        # best_model = model_train(model, optimizer, scheduler, train_loader, val_loader, max_epochs, device)
        today = date.today()
        n_log = len([f for f in os.listdir(log_dir) if
                     os.path.isfile(os.path.join(log_dir, f)) and today.strftime("%b-%d-%Y") in f])
        log_fn = log_dir + today.strftime("%b-%d-%Y") + '_' + str(n_log) + '.log'
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(
            "Experiment {} on date {}. Decade: {}.".format(n_log, today.strftime("%b-%d-%Y"), decade) +
            " Loss function type: {}. Training Batch size: {}.".format(loss_function, train_batch_size) +
            " Evaluation Batch size: {}. Learning rate: {}. Modalities:{}. Joint hidden dim: {}."
            .format(val_batch_size, learning_rate, modalities, hidden_dim_2))

        # # evaluate the initialized model
        # eval_outputs = model_evaluate(model, val_loader, device, loss_function, max_epochs_eval)
        # logging.info(
        #     "Decade {}, initialized model,".format(decade) +
        #     " evaluation accuracy: {},".format(eval_outputs['acc']) +
        #     " mean evaluation rank: {}".format(eval_outputs['rank'])
        # )

        best_eval_auc = 0
        best_epoch = 0
        best_model = deepcopy(model)
        best_test_results = {}
        for epoch in tqdm(range(max_epochs_train), position=0, leave=True):
            model.train()
            training_loss = 0
            for episode, (frame_idx_tensor, query_nouns_idx_tensor, support_nouns_idx_tensor,
                          support_nouns_total_counts_tensor) in tqdm(enumerate(train_loader), position=0, leave=True):

                query_nouns_idx_tensor, support_nouns_idx_tensor, support_nouns_total_counts_tensor \
                    = query_nouns_idx_tensor.to(device), support_nouns_idx_tensor.to(device), \
                      support_nouns_total_counts_tensor.to(device)
                Hq_fused = model(query_nouns_idx_tensor)
                Hs_fused = model(support_nouns_idx_tensor)
                if loss_function == 'proto_loss':
                    output = proto_loss(Hq_fused, Hs_fused)
                else:
                    output = exemplar_loss(Hq_fused, Hs_fused)
                loss = output['loss']
                training_loss += loss.item()
                scaler.scale(loss).backward()
                if (episode + 1) % iters_to_accumulate == 0:
                    # Adjust the learning rate based on the number of iterations.
                    scheduler.step()
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()
                    # Clear gradients
                    optimizer.zero_grad()
            logging.info(
                "Decade {} Epoch {}, training loss: {},".format(decade, epoch + 1, training_loss / len(train_ds)))

            # evaluate model at the end of every 'eval_every' epochs
            if (epoch + 1) % eval_every == 0:
                test_results = model_test(model, noun_test_loader, frame_test_loader, device, decade_co_occurrence_mat,
                                          loss_function, topk=20)
                if test_results['auc_frame'] > best_eval_auc:
                    best_eval_auc = test_results['auc_frame']
                    best_epoch = epoch
                    best_model = deepcopy(model)
                    best_test_results = test_results
                logging.info(
                    "Decade {} Epoch {},".format(decade, epoch + 1) +
                    " mean auc score: {},".format(test_results['auc_frame'])
                )
        # log best evaluation results
        logging.info(
            "Decade {}, best evaluation auc score: {}, at epoch {}".format(
                decade, best_eval_auc,
                best_epoch))
        # save prediction results for the last epoch
        print('saving prediction results...')
        best_pred_df_noun = pd.DataFrame(best_test_results['noun_prediction_df'])
        best_pred_df_frame = pd.DataFrame(best_test_results['frame_prediction_df'])

        best_pred_df_noun.to_csv(prediction_dir + '{}_{}_{}_noun.csv'.format(decade, loss_function,
                                                                             '-'.join(modalities)))
        best_pred_df_frame.to_csv(prediction_dir + '{}_{}_{}_frame.csv'.format(decade, loss_function,
                                                                               '-'.join(modalities)))

        # save final model parameters
        model_fn = '{}_{}_{}_best.pt'.format(decade, loss_function, '-'.join(modalities))
        torch.save(best_model.state_dict(), model_dir + model_fn)
