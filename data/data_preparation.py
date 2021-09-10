from gsn import *
from conceptnet import *
from imagenet import *
import pickle


def main():
    # get gsn statistics of word counts
    yearly_abs_counts, yearly_percent_counts, gsn_word2idx = get_word_counts(start_year=1800, end_year=2000,
                                                                             min_total_count=20000)

    # get conceptnet statistics of word counts and edge weights
    gsn_vocab = gsn_word2idx.keys()
    word2cnp_counts, cnp_word2idx, weighted_co_occurrences = get_cnp_stats(gsn_vocab)

    # get mappings between ImageNet index, WordNetID (wnid) and words
    image_paths, wnids = get_img_paths()
    wnid2img_path_idx = get_wnid2img_path_idx(image_paths)
    word2wnids = get_word2wnids(wnids)
    word2img_idx = get_word2img_idx(word2wnids, wnid2img_path_idx)

    # get words with sufficient ontological and visual representations
    min_cnp_edges = 10
    min_img_num = 100
    cnp_vocab = set(cnp_word2idx.keys())
    img_vocab = set(word2img_idx.keys())
    cnp_img_common_vocab = set([word for word in cnp_vocab if word in img_vocab
                                and word2cnp_counts[word] >= min_cnp_edges
                                and len(word2img_idx[word]) >= min_img_num])

    # get words with sufficient GSN occurrences
    word2gsn_idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/word2idx.p', 'rb'))
    gsn_yearly_abs_counts = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/yearly_abs_counts.p', 'rb'))
    gsn_vocab = set(word2gsn_idx.keys())
    gsn_word2total_count_by_pos = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/word2total_count.p', 'rb'))

    freq_threshold = 100000
    candidate_support_nouns = set([word for word in cnp_img_common_vocab
                                   if sum(gsn_word2total_count_by_pos[word]) >= freq_threshold
                                   and word in gsn_vocab
                                   ])

    # choose candidate verbs with sufficient frequency in GSN
    min_verb_freq = 60000
    candidate_verbs = [word for word in word2gsn_idx.keys() if
                       gsn_word2total_count_by_pos[word][1] >= min_verb_freq]

    # extract v-r-n frame usages from GSN
    rel_types = {'nsubj', 'dobj', 'iobj', 'pobj'}
    preps = {'in', 'by', 'to', 'with', 'on', 'from', 'for', 'at', 'as', 'like', 'of', 'into', 'about', 'under'}
    frame_df = get_frame_data(candidate_support_nouns, candidate_verbs, rel_types, preps)
    frame_df.to_csv('/h/19/jadeleiyu/frame_extension/data/gsn/gsn_frame_df.csv', index=False)

    # aggregate frame usages with the same (noun, verb, relation) triples
    agg_func = {}
    for decade in range(1800, 2010, 10):
        agg_func['count in {}s'.format(decade)] = sum
    grouped_frame_df = frame_df.groupby(['verb', 'relation', 'noun']).agg(agg_func).reset_index()

    # aggregate frame usages by their support nouns
    agg_func = {'noun': lambda x: list(x)}
    for decade in range(1800, 2010, 10):
        agg_func['count in {}s'.format(decade)] = lambda x: list(x)
    grouped_frame_df = grouped_frame_df.groupby(['verb', 'relation']).agg(agg_func).reset_index()

    # compute total counts for each frame usage up to each decade from 1800s to 2000s
    decade2total_counts_arr = {}
    for decade in range(1800, 2010, 10):
        decade2total_counts_arr[decade] = []
    for index, row in grouped_frame_df.iterrows():
        num_nouns = len(row['noun'])
        decade_counts = np.zeros((21, num_nouns))
        for decade in range(1800, 2010, 10):
            decade_idx = int((decade - 1800) / 10)
            col_name = "count in {}s".format(decade)
            decade_counts[decade_idx] = np.array(row[col_name], dtype=int)
            total_counts = decade_counts[:decade_idx].sum(axis=0).astype(int)
            decade2total_counts_arr[decade].append(total_counts)
    for decade in range(1800, 2010, 10):
        grouped_frame_df['total counts up to {}s'.format(decade)] = decade2total_counts_arr[decade]

    # get the counts for each support noun by decade in frame_df
    # so that when computing historical conceptnet embeddings later, we weight each node by its frequency on every decade
    # in this way we can effectively avoid the affect of OCR-like errors
    # e.g. even if the word "car" is present in GSN at 1800s, its extremely low frequency will make it contribute very
    # few to the diachronic embeddings at 1800s
    agg_func = {}
    for decade in range(1800, 2010, 10):
        agg_func['count in {}s'.format(decade)] = sum
    noun_grouped_frame_df = frame_df.groupby(['noun']).agg(agg_func).reset_index()
    noun_grouped_frame_df.to_csv('/h/19/jadeleiyu/frame_extension/data/gsn/noun_decade_counts_df.csv', index=False)

    # compute numpy array of decade counts for support nouns
    support_nouns = list(noun_grouped_frame_df['noun'])
    support_noun2idx = {support_nouns[i]: i for i in range(len(support_nouns))}

    noun_decade_counts = np.zeros((21, len(support_nouns)))
    for decade in range(1800, 2010, 10):
        decade_idx = int((decade - 1800) / 10)
        noun_decade_counts[decade_idx] = np.array(noun_grouped_frame_df['count in {}s'.format(decade)])

    pickle.dump(noun_decade_counts.astype(int),
                open('/h/19/jadeleiyu/frame_extension/data/gsn/noun_decade_counts.p', 'wb'))
    pickle.dump(support_noun2idx, open('/h/19/jadeleiyu/frame_extension/data/gsn/support_noun2idx.p', 'wb'))

    # compute conceptnet historical representations for all words in cnp vocab
    # later we will only make use of the embeddings of support nouns
    compute_cnp_hist_embeddings(support_noun2idx, cnp_word2idx, weighted_co_occurrences, noun_decade_counts)

    # compute imagenet representations for support nouns
    compute_visual_representations(support_noun2idx, image_paths, word2img_idx, img_sample_size=64)
