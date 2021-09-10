import numpy as np
import pickle


def get_histwords_embeddings(noun2idx, start_year=1800, end_year=1990):
    for decade in range(start_year, end_year + 10, 10):
        histwords_embs_unalinged = np.load(
            '/hal9000/datasets/wordembeddings/historical_sgns_all_english/{}-w.npy'.format(decade))
        histwords_vocab = pickle.load(
            open('/hal9000/datasets/wordembeddings/historical_sgns_all_english/{}-vocab.pkl'.format(decade), 'rb'))
        histwords_vocab2idx = {w: i for (i, w) in enumerate(histwords_vocab)}
        N = len(noun2idx)
        embedding_dim = histwords_embs_unalinged.shape[-1]
        noun_embs_matrix = np.zeros((N, embedding_dim))
        for noun in noun2idx.keys():
            aligned_idx = noun2idx[noun]
            if noun in histwords_vocab2idx.keys():
                unaligned_idx = histwords_vocab2idx[noun]
                emb = histwords_embs_unalinged[unaligned_idx]
                noun_embs_matrix[aligned_idx] = emb
            else:
                noun_embs_matrix[aligned_idx] = np.random.rand(embedding_dim)
        np.save('/h/19/jadeleiyu/frame_extension/data/histwords_embeddings/embeddings_{}.npy'.format(decade),
                noun_embs_matrix)