import pickle
from collections import namedtuple

import pandas as pd
from torch.utils.data import DataLoader
from dataset import GSNFrameDataset
from evaluation import model_test
import torch
import numpy as np
from models import LikelihoodNetwork

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
    noun_decade_counts = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/noun_decade_counts.p', 'rb'))
    list_converter = {'novel nouns': eval, 'most similar novel nouns': eval, 'least similar novel nouns': eval,
                      'established nouns': eval, 'ground truth extended frames': eval,
                      'support nouns': eval, 'ground truth novel nouns': eval, 'support noun counts': eval,
                      'established noun counts': eval}

    count_thresholds = np.array([0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    cuda_available = torch.cuda.is_available()
    if cuda and cuda_available:
        device = torch.device("cuda")
        print("using gpu acceleration")
    else:
        device = torch.device("cpu")
        print("using cpu")

    mean_precisions = []
    for decade in range(start_decade, end_decade + 10, 10):
        decade_idx = int((decade - 1800) / 10)
        print('begin learning in decade {}s'.format(decade))
        print('preparing learning data...')

        decade_noun_test_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/noun_eval_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_frame_test_df = pd.read_csv(
            '/h/19/jadeleiyu/frame_extension/data/gsn/frame_eval_df_{}s.csv'.format(decade),
            converters=list_converter)
        decade_co_occurrence_mat = torch.load('/h/19/jadeleiyu/frame_extension/data/gsn/co_occurrence_mat_{}s.pt'
                                              .format(decade))
        noun_counts = noun_decade_counts[decade_idx]
        noun2decade_count = {n: noun_counts[idx] for (n, idx) in noun2idx.items()}

        noun_test_ds = GSNFrameDataset(decade_noun_test_df, noun2idx, n_q, n_s, learning_type='test_noun')
        frame_test_ds = GSNFrameDataset(decade_frame_test_df, noun2idx, n_q, n_s, learning_type='test_frame')

        noun_test_loader = DataLoader(noun_test_ds, shuffle=False, batch_size=test_batch_size, num_workers=num_workers)
        frame_test_loader = DataLoader(frame_test_ds, shuffle=False, batch_size=test_batch_size,
                                       num_workers=num_workers)

        conceptnet_embeddings = torch.tensor(pickle.load(
            open('/h/19/jadeleiyu/frame_extension/data/cnp/cnp_hist_embeddings_{}'.format(decade), 'rb')))
        histwords_embeddings = torch.tensor(np.load(
            '/h/19/jadeleiyu/frame_extension/data/histwords_embeddings/embeddings_{}.npy'.format(decade)))
        imagenet_embeddings = torch.load('/h/19/jadeleiyu/frame_extension/data/img/noun_image_features.pt')

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

        model = LikelihoodNetwork(config).to(device)
        model_fn = '{}_{}_{}_best.pt'.format(decade, loss_function, '-'.join(modalities))
        model.load_state_dict(torch.load(model_dir + model_fn))

        test_results = model_test(model, noun_test_loader, frame_test_loader, device, decade_co_occurrence_mat,
                                  loss_function, topk=20)
        precisions_f = test_results['precisions_frame'].mean(axis=0)
        mean_precisions.append(precisions_f)

        # query_nouns = list(decade_noun_test_df['query noun'])
        # query_noun_decade_counts = [noun2decade_count[noun] for noun in query_nouns]
        # decade_noun_test_df['query noun acc count'] = pd.Series(query_noun_decade_counts)
        # decade_noun_test_df['query noun precision'] = pd.Series(precisions_f)
        # decade_mean_precisions = \
        #     decade_noun_test_df.groupby(pd.cut(decade_noun_test_df['query noun acc count'], count_thresholds)).mean()[
        #         'query noun precision']
        # mean_precisions.append(decade_mean_precisions)
        pickle.dump(test_results['Hqs'],
                    open(
                        prediction_dir + 'Hqs_{}_{}_{}'.format(decade, loss_function, '-'.join(modalities)),
                        'wb')
                    )
        pickle.dump(test_results['Hss'],
                    open(
                        prediction_dir + 'Hss_{}_{}_{}'.format(decade, loss_function, '-'.join(modalities)),
                        'wb')
                    )
    # save noun-wise precisions over decades
    pickle.dump(mean_precisions,
                open(prediction_dir + 'mean_precisions_{}_{}'.format(loss_function, '-'.join(modalities)), 'wb'))




def tsne_plots(decade):
    decade_idx = int((decade - 1800) / 10)
    noun2idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/support_noun2idx.p',
                                'rb'))
    list_converter = {'novel nouns': eval, 'most similar novel nouns': eval, 'least similar novel nouns': eval,
                      'established nouns': eval, 'ground truth extended frames': eval,
                      'support nouns': eval, 'ground truth novel nouns': eval, 'support noun counts': eval,
                      'established noun counts': eval}
    prediction_dir = '/h/19/jadeleiyu/frame_extension/predictions/'

    mean_precisions_ling = pickle.load(open(prediction_dir + 'mean_precisions_exemplar_loss_ling', 'rb'))[decade_idx]
    mean_precisions_vo = pickle.load(open(prediction_dir + 'mean_precisions_exemplar_loss_vis_ont', 'rb'))[decade_idx]

    decade_noun_test_df = pd.read_csv(
        '/h/19/jadeleiyu/frame_extension/data/gsn/noun_eval_df_{}s.csv'.format(decade),
        converters=list_converter)
    decade_frame_test_df = pd.read_csv(
        '/h/19/jadeleiyu/frame_extension/data/gsn/frame_eval_df_{}s.csv'.format(decade),
        converters=list_converter)

    Hqs_ling = pickle.load(open(prediction_dir + 'Hqs_{}_exemplar_loss_ling'.format(decade), 'rb'))
    Hss_ling = pickle.load(open(prediction_dir + 'Hss_{}_exemplar_loss_ling'.format(decade)))

    Hqs_vo = pickle.load(open(prediction_dir + 'Hqs_{}_exemplar_loss_vis_ont'.format(decade), 'rb'))
    Hss_vo = pickle.load(open(prediction_dir + 'Hss_{}_exemplar_loss_vis_ont'.format(decade), 'rb'))



