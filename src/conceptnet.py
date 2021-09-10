import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm


def get_cnp_stats(gsn_vocab, assertion_csv_file='/h/19/jadeleiyu/frame_extension/data/cnp/assertions.csv',
                  chunksize=10000):
    try:
        word2cnp_counts = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/cnp/word2cnp_counts.p', 'rb'))
        word2idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/cnp/word2cnp_idx.p', 'rb'))
        weighted_co_occurrences = pickle.load(
            open('/h/19/jadeleiyu/frame_extension/data/cnp/weighted_co_occurrences.p', 'rb'))
    except FileNotFoundError:
        word2cnp_counts = {}
        weighted_co_occurrences = []
        lemmatizer = WordNetLemmatizer()
        for assertions_df in tqdm(pd.read_csv(assertion_csv_file, header=None, delimiter='\t',
                                              chunksize=chunksize, error_bad_lines=False)):
            for index, row in assertions_df.iterrows():
                start_node_list = row[2].split('/')
                end_node_list = row[3].split('/')
                start_node_lang = start_node_list[2]
                end_node_lang = end_node_list[2]
                if len(start_node_list) > 4:
                    start_node_pos = start_node_list[4]
                    if len(end_node_list) > 4:
                        end_node_pos = end_node_list[4]
                    else:
                        end_node_pos = start_node_pos
                    if start_node_lang == 'en' and end_node_lang == 'en':
                        start_node_word = lemmatizer.lemmatize(start_node_list[3], pos=start_node_pos)
                        end_node_word = lemmatizer.lemmatize(end_node_list[3], pos=end_node_pos)
                        weight = literal_eval(row[4])['weight']
                        weighted_co_occurrences.append(
                            ('_'.join([start_node_word, start_node_pos]), '_'.join([end_node_word, end_node_pos]),
                             weight))
                        if start_node_word not in word2cnp_counts:
                            word2cnp_counts[start_node_word] = 1
                        else:
                            word2cnp_counts[start_node_word] += 1
        vocab = set(word2cnp_counts.keys()).intersection(gsn_vocab)
        word2idx = {w: i for (i, w) in enumerate(vocab)}

        pickle.dump(word2cnp_counts, open('/h/19/jadeleiyu/frame_extension/data/cnp/word2cnp_counts.p', 'wb'))
        pickle.dump(word2idx, open('/h/19/jadeleiyu/frame_extension/data/cnp/word2cnp_idx.p', 'wb'))
        pickle.dump(weighted_co_occurrences,
                    open('/h/19/jadeleiyu/frame_extension/data/cnp/weighted_co_occurrences.p', 'wb'))
    return word2cnp_counts, word2idx, weighted_co_occurrences


def counts_to_ppmi(counts_csr, smoothing=0.75):
    """
    Converts a sparse matrix of co-occurrences into a sparse matrix of positive
    pointwise mutual information. Context distributional smoothing is applied
    to the resulting matrix.
    """
    # word_counts adds up the total amount of association for each term.
    word_counts = np.asarray(counts_csr.sum(axis=1)).flatten()

    # smooth_context_freqs represents the relative frequency of occurrence
    # of each term as a context (a column of the table).
    smooth_context_freqs = np.asarray(counts_csr.sum(axis=0)).flatten() ** smoothing
    smooth_context_freqs /= smooth_context_freqs.sum()

    # Divide each row of counts_csr by the word counts. We accomplish this by
    # multiplying on the left by the sparse diagonal matrix of 1 / word_counts.
    ppmi = sparse.diags(1 / word_counts).dot(counts_csr)

    # Then, similarly divide the columns by smooth_context_freqs, by the same
    # method except that we multiply on the right.
    ppmi = ppmi.dot(sparse.diags(1 / smooth_context_freqs))

    # Take the log of the resulting entries to give pointwise mutual
    # information. Discard those whose PMI is less than 0, to give positive
    # pointwise mutual information (PPMI).
    ppmi.data = np.maximum(np.log(ppmi.data), 0)
    ppmi.eliminate_zeros()
    return ppmi


def align_hist_embeddings(base_embeddings, noun_embeddings):
    # m = other_vecs.T.dot(base_vecs)
    # # SVD method from numpy
    # u, _, v = np.linalg.svd(m)
    #
    # ortho = u.dot(v)
    # # Replace original array with modified one
    # # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    # other_embed.wv.vectors_norm = other_embed.wv.syn0 = other_embed.wv.vectors_norm.dot(ortho)
    # return other_embed
    m = noun_embeddings.T.dot(base_embeddings)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)
    return noun_embeddings.dot(ortho)


def compute_cnp_hist_embeddings(support_noun2idx, word2cnp_idx, weighted_co_occurrences, noun_decade_counts,
                                embedding_dim=300, start_year=1800,
                                end_year=2000):
    for decade in tqdm(range(start_year, end_year + 10, 10)):
        try:
            embeddings = pickle.load(
                open('/h/19/jadeleiyu/frame_extension/data/cnp/cnp_hist_embeddings_{}'.format(decade), 'rb'))
        except FileNotFoundError:
            base_embeddings = pickle.load(
                open('/h/19/jadeleiyu/frame_extension/data/cnp/cnp_hist_embeddings_{}'.format(1800), 'rb'))
            decade_idx = int((decade - 1800) / 10)
            noun_decade_weights = noun_decade_counts[decade_idx] / np.sum(noun_decade_counts[decade_idx])
            M = len(support_noun2idx)
            N = len(word2cnp_idx)
            co_occurrence_mat = np.zeros((M, N))
            for (start_node, end_node, weight) in weighted_co_occurrences:
                w_s = start_node.split('_')[0]
                w_e = end_node.split('_')[0]
                if w_s in support_noun2idx.keys() and w_e in word2cnp_idx.keys():
                    row_idx = support_noun2idx[w_s]
                    col_idx = word2cnp_idx[w_e]
                    co_occurrence_mat[row_idx][col_idx] += weight * noun_decade_weights[support_noun2idx[w_s]]
            print('converting co-occurrence matrix of decade {} into PPMI matrix...'.format(decade))
            # convert co-occurrence matrix into PPMI matrix
            ppmi_mat = counts_to_ppmi(sparse.csr_matrix(co_occurrence_mat))

            print("performing SVD on PPMI matrix of decade {}...".format(decade))
            # perform truncated SVD to obtain embeddings of dim 300
            svd = TruncatedSVD(n_components=embedding_dim, n_iter=10, random_state=42)
            embeddings = svd.fit_transform(ppmi_mat)
            if decade == 1800:
                base_embeddings = embeddings
            else:
                embeddings = align_hist_embeddings(base_embeddings, embeddings)
            pickle.dump(embeddings,
                        open('/h/19/jadeleiyu/frame_extension/data/cnp/cnp_hist_embeddings_{}'.format(decade), 'wb'))
