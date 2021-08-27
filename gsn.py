import gzip
import pickle
import random

import numpy as np
import pandas as pd
import requests
import ruptures as rpt
import torch
from nltk import WordNetLemmatizer
from tqdm import tqdm
from ast import literal_eval

ROOT_DIR = '/h/19/jadeleiyu/frame_extension/'


def pos2k(pos):
    if pos:
        if pos[0] == 'N':
            return 0
        elif pos[0] == 'V':
            return 1
        elif pos[0] == 'J':
            return 2
        elif pos[0] == 'R':
            return 3
        else:
            return 4
    return 4


def rel_extract(tagged_words, noun_idx, rel_types, preps):
    # as/IN/mark/3 water/NN/nsubj/3 covers/VBZ/advcl/0 the/DT/det/5 sea/NN/dobj/3
    # ['as/IN/mark/3', 'water/NN/nsubj/3', 'covers/VBZ/advcl/0',
    # 'for/IN/prep/3', 'the/DT/det/5', 'sea/NN/dobj/4']
    relation = []
    # words = [tagged_word.split('/')[0] for tagged_word in tagged_words]
    current_idx = noun_idx
    current_node = tagged_words[current_idx]
    next_idx = int(current_node.split('/')[-1]) - 1
    while next_idx != -1:
        rel = current_node.split('/')[-2]
        prep = current_node.split('/')[0]
        if rel in rel_types:
            relation.append(current_node.split('/')[-2])
        elif rel == 'prep' and prep in preps:
            relation.append('.'.join([current_node.split('/')[-2], current_node.split('/')[0]]))
        else:
            return None
        current_idx = next_idx
        current_node = tagged_words[current_idx]
        next_idx = int(current_node.split('/')[-1]) - 1
    if len(relation) == 1:
        return relation[0]
    elif len(relation) == 2:
        return '_'.join(relation)
    return None


def get_word_counts(start_year=1800, end_year=2000, min_total_count=20000):
    """
    Compute total counts (by POS) and yearly counts for words in GSN with total frequency above a given threshold.
    V: size of vocab
    N_year: length of the time span of study
    :return:
    total_counts: an np array of shape (V, 5), where the i-th row is the total counts of word w_i of 5 different POS tags.
    yearly_abs_counts: an np array of shape (V, N_year), where each row is the yearly count for word w_i from 1800s to 2000s.
    yearly_percent_counts: an np array of shape (V, N_year), where each row is the percentage of yearly count
        (i.e. (total count of w_i at time t)/(total word count in GSN at time t))
        for word w_i from 1800s to 2000s.
    word2idx: a lookup dict that maps each word to its row index in total_counts and yearly_counts
    """
    try:
        total_counts = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn_word_total_counts.p', 'rb'))
        yearly_abs_counts = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn_word_yearly_abs_counts.p', 'rb'))
        yearly_percent_counts = pickle.load(
            open('/h/19/jadeleiyu/frame_extension/data/gsn_word_yearly_percent_counts.p', 'rb'))
        word2idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn_word2idx.p', 'rb'))

    except FileNotFoundError:
        try:
            word2total_count = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/word2total_count.p', 'rb'))
            word2yearly_count = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/gsn/word2yearly_count.p', 'rb'))
        except FileNotFoundError:
            lemmatizer = WordNetLemmatizer()
            word2total_count = {}
            word2yearly_count = {}
            for i in tqdm(range(99)):
                node_id = str(i).rjust(2, '0')
                url = 'http://commondatastorage.googleapis.com/books/syntactic-ngrams/eng/nodes.' + node_id + '-of-99.gz'
                r = requests.get(url)
                content = gzip.decompress(r.content)
                for line in content.decode("utf-8").split('\n')[:-1]:
                    line_stats = line.split('\t')
                    pos = line_stats[1].split('/')[1]
                    k = pos2k(pos)
                    if k == 0:
                        head_word = lemmatizer.lemmatize(line_stats[0], pos='n')
                    elif k == 1:
                        head_word = lemmatizer.lemmatize(line_stats[0], pos='v')
                    else:
                        head_word = line_stats[0]
                    if head_word.isalpha():
                        counts_by_year = line_stats[3:]
                        total_count = int(line_stats[2])
                        if head_word not in word2total_count:
                            word2total_count[head_word] = [0] * 5
                            word2total_count[head_word][k] = total_count
                        else:
                            word2total_count[head_word][k] += total_count
                        if head_word not in word2yearly_count:
                            word2yearly_count[head_word] = [0] * (end_year - start_year + 1)
                            for yearly_count in counts_by_year:
                                year = int(yearly_count.split(',')[0])
                                if start_year <= year <= end_year:
                                    idx = year - start_year
                                    count = int(yearly_count.split(',')[1])
                                    word2yearly_count[head_word][idx] = count
                        else:
                            for yearly_count in counts_by_year:
                                year = int(yearly_count.split(',')[0])
                                if start_year <= year <= end_year:
                                    idx = year - start_year
                                    count = int(yearly_count.split(',')[1])
                                    word2yearly_count[head_word][idx] += count
            pickle.dump(word2yearly_count, open('/h/19/jadeleiyu/frame_extension/data/gsn/word2yearly_count.p', 'wb'))
            pickle.dump(word2total_count, open('/h/19/jadeleiyu/frame_extension/data/gsn/word2total_count.p', 'wb'))

        vocab = []
        yearly_abs_counts = []
        for word, yearly_counts in word2yearly_count.items():
            if sum(yearly_counts) >= min_total_count:
                vocab.append(word)
                yearly_abs_counts.append(np.array(word2yearly_count[word]))
        yearly_abs_counts = np.array(yearly_abs_counts)
        total_counts_by_year = np.tile(yearly_abs_counts.sum(axis=0), (len(vocab), 1))  # sum over cols: (V, T) --> (T)
        yearly_percent_counts = np.multiply(yearly_abs_counts.astype(float),
                                            np.reciprocal(total_counts_by_year.astype(float)))
        word2idx = {w: i for (i, w) in enumerate(vocab)}

        pickle.dump(yearly_abs_counts, open('/h/19/jadeleiyu/frame_extension/data/gsn/yearly_abs_counts.p', 'wb'))
        pickle.dump(yearly_percent_counts,
                    open('/h/19/jadeleiyu/frame_extension/data/gsn/yearly_percent_counts.p', 'wb'))
        pickle.dump(word2idx, open('/h/19/jadeleiyu/frame_extension/data/gsn/word2idx.p', 'wb'))
    return yearly_abs_counts, yearly_percent_counts, word2idx


def get_word2toe(yearly_percent_counts, word2idx, words, model='rbf', pen=5, jump=1, qunatile_threshold=-2,
                 start_year=1800, end_year=2000):
    word2toe = {}
    yearly_percent_means = np.mean(yearly_percent_counts, axis=0)
    yearly_percent_vars = np.var(yearly_percent_counts, axis=0)
    for word in tqdm(words):
        percent_ts = yearly_percent_counts[word2idx[word]]
        algo = rpt.Pelt(model=model, jump=jump).fit(percent_ts)
        change_points = algo.predict(pen=pen)
        cp0 = change_points[0]
        if cp0 <= end_year - start_year:
            z_percent = (percent_ts[cp0] - yearly_percent_means[cp0]) / yearly_percent_vars[cp0]
            if z_percent <= qunatile_threshold:
                word2toe[word] = change_points[0] + start_year
            else:
                word2toe[word] = start_year
        else:
            word2toe[word] = start_year
    return word2toe


def ngram_lemmatize(lemmatizer, tagged_word):
    if tagged_word.split('/')[1]:
        if tagged_word.split('/')[1][0] == 'V':
            return lemmatizer.lemmatize(tagged_word.split('/')[0], pos='v'), tagged_word.split('/')[1]
        elif tagged_word.split('/')[1][0] == 'N':
            return lemmatizer.lemmatize(tagged_word.split('/')[0], pos='n'), tagged_word.split('/')[1]
        else:
            return tagged_word.split('/')[0], tagged_word.split('/')[1]


def get_frame_data(candidate_nouns, candidate_verbs, rel_types, preps, start_year=1800, end_year=2000):
    lemmatizer = WordNetLemmatizer()
    frame_df = {
        'verb': [],
        'relation': [],
        'noun': [],
    }

    for decade in range(start_year, end_year + 10, 10):
        col_name = 'count in {}s'.format(str(decade))
        frame_df[col_name] = []
    N = 0
    n_error = 0
    for i in tqdm(range(99)):
        node_id = str(i).rjust(2, '0')
        url = 'http://commondatastorage.googleapis.com/books/syntactic-ngrams/eng/verbargs.' \
              + node_id + '-of-99.gz'
        r = requests.get(url)
        content = gzip.decompress(r.content)
        for line in tqdm(content.decode("utf-8").split('\n'), position=0, leave=True):
            N += 1
            try:
                line_stats = line.split('\t')
                head_verb = lemmatizer.lemmatize(line_stats[0], pos='v')
                if head_verb in candidate_verbs:
                    syntactic_ngram = line_stats[1]
                    tagged_words = syntactic_ngram.split(' ')
                    words_with_pos = [ngram_lemmatize(lemmatizer, tagged_word) for tagged_word in
                                      tagged_words]
                    if words_with_pos:
                        nouns = [word for (word, pos) in words_with_pos if
                                 word in candidate_nouns and pos[0] == 'N']
                        if nouns:
                            noun = nouns[0]
                            words = [t[0] for t in words_with_pos]
                            noun_idx = words.index(noun)
                            relation = rel_extract(tagged_words, noun_idx, rel_types, preps)
                            if relation:
                                frame_df['verb'].append(head_verb)
                                frame_df['noun'].append(noun)
                                frame_df['relation'].append(relation)
                                counts_by_decade = [0] * 21
                                for x in line_stats[3:]:
                                    year = int(x.split(',')[0])
                                    if year >= start_year:
                                        decade_idx = int((year - start_year) / 10)
                                        count = int(x.split(',')[1])
                                        counts_by_decade[decade_idx] += count
                                for k in range(len(counts_by_decade)):
                                    decade = start_year + k * 10
                                    col_name = 'count in {}s'.format(str(decade))
                                    frame_df[col_name].append(counts_by_decade[k])
            except Exception as e:
                n_error += 1
                pass
        print('file {} with {} lines in total, get {} errors'.format(i, N, n_error))
    frame_df = pd.DataFrame(frame_df)
    return frame_df


def learning_df_prep(grouped_frame_df, start_dec=1980, end_dec=2000, min_frame_total_count=2000,
                     min_established_noun_total_count=200, min_novel_noun_final_total_count=100,
                     max_novel_noun_current_count=20, n_s=8, n_q_train=4, n_q_eval=1):
    for decade in range(start_dec, end_dec + 10, 10):
        training_df = {
            'frame': [],
            'frame id': [],
            'established nouns': [],
            'established noun counts'.format(decade): [],
            'novel nouns': [],
            'novel nouns train': [],
            'novel nouns evaluation': []
        }

        for index, row in tqdm(grouped_frame_df.iterrows()):
            final_decade_acc_counts = eval(row["total count up to {}s".format(2000)])
            frame_total_count = sum(final_decade_acc_counts)
            if frame_total_count >= min_frame_total_count:
                nouns = eval(row['noun'])
                estab_nouns = []
                estab_noun_counts = []
                novel_nouns = []
                novel_noun_counts = []
                cur_decade_acc_counts = eval(row["total count up to {}s".format(decade)])
                for j in range(len(nouns)):
                    if (max_novel_noun_current_count >= cur_decade_acc_counts[j]) and (
                            final_decade_acc_counts[j] >= min_novel_noun_final_total_count):
                        novel_nouns.append(nouns[j])
                        novel_noun_counts.append(final_decade_acc_counts[j])
                    elif cur_decade_acc_counts[j] >= min_established_noun_total_count:
                        estab_nouns.append(nouns[j])
                        estab_noun_counts.append(cur_decade_acc_counts[j])
                if len(estab_nouns) >= n_s and len(novel_nouns) >= n_q_train + n_q_eval:
                    if len(estab_nouns) > n_s:
                        max_estab_nouns_ids = np.argsort(-np.array(estab_noun_counts))[:n_s]
                        estab_nouns = [estab_nouns[i] for i in max_estab_nouns_ids]
                        estab_noun_counts = [estab_noun_counts[i] for i in max_estab_nouns_ids]
                    if len(novel_nouns) > n_q_train + n_q_eval:
                        max_novel_nouns_ids = np.argsort(-np.array(novel_noun_counts))[:n_q_train + n_q_eval]
                        novel_nouns = [novel_nouns[i] for i in max_novel_nouns_ids]
                    frame = '-'.join([row['verb'], row['relation']])
                    training_df['frame'].append(frame)
                    training_df['established nouns'].append(estab_nouns)
                    training_df['established noun counts'].append(estab_noun_counts)
                    random.shuffle(novel_nouns)
                    training_df['novel nouns'].append(novel_nouns)
                    training_df['novel nouns train'].append(novel_nouns[:n_q_train])
                    training_df['novel nouns evaluation'].append(novel_nouns[n_q_train:])

        training_df['frame id'] = pd.Series(list(range(len(training_df['frame']))))
        training_df = pd.DataFrame(training_df)
        training_df.to_csv('/h/19/jadeleiyu/frame_extension/data/gsn/training_df_{}s.csv'.format(decade),
                           index=False)


def similarity_filtering(training_df, noun2idx, vis_embeddings, ont_embeddings, topk=8):
    pos_queries = []
    neg_queries = []
    for index, row in training_df.iterrows():
        support_nouns_idx = torch.tensor([noun2idx[noun] for noun in row['established nouns']])
        query_nouns_idx = torch.tensor([noun2idx[noun] for noun in row['novel nouns']])
        E_s = torch.cat([vis_embeddings[support_nouns_idx], ont_embeddings[support_nouns_idx]], dim=-1)
        E_q = torch.cat([vis_embeddings[query_nouns_idx], ont_embeddings[query_nouns_idx]], dim=-1)
        euc_dists = torch.cdist(E_q, E_s).sum(dim=-1)
        pos_queries_idx = torch.argsort(euc_dists)[:topk].tolist()
        neg_queries_idx = torch.argsort(-euc_dists)[:topk].tolist()
        row_pos_queries = [row['novel nouns'][k] for k in pos_queries_idx]
        row_neg_queries = [row['novel nouns'][k] for k in neg_queries_idx]
        pos_queries.append(row_pos_queries)
        neg_queries.append(row_neg_queries)
    return pos_queries, neg_queries


def get_test_dfs(start_decade=1980, end_decade=2000):
    for decade in range(start_decade, end_decade+10, 10):
        training_df = pd.read_csv('/h/19/jadeleiyu/frame_extension/data/gsn/training_df_{}s.csv'.format(decade))
        query_noun_eval_df = {
            'query noun': [],
            'ground truth extended frames': []
        }

        frame_eval_df = {
            'frame': training_df['frame'],
            'support nouns': training_df['established nouns'],
            'support noun counts': training_df['established noun counts'],
            'ground truth novel nouns': training_df['novel nouns'],
        }
        for index, row in training_df.iterrows():
            novel_nouns = eval(row['novel nouns'])
            for novel_noun in novel_nouns:
                query_noun_eval_df['query noun'].append(novel_noun)
                query_noun_eval_df['ground truth extended frames'].append(row['frame'])
        query_noun_eval_df = pd.DataFrame(query_noun_eval_df)
        agg_func = {'ground truth extended frames': lambda x: list(x)}
        query_noun_eval_df = query_noun_eval_df.groupby(['query noun']).agg(agg_func).reset_index()
        frame_eval_df = pd.DataFrame(frame_eval_df)

        N_nouns = len(query_noun_eval_df['query noun'])
        N_frames = len(frame_eval_df['frame'])
        frame2idx = {row['frame']: index for (index, row) in frame_eval_df.iterrows()}
        co_occurrence_mat = torch.zeros(N_nouns, N_frames)
        for index, row in query_noun_eval_df.iterrows():
            for frame in row['ground truth extended frames']:
                co_occurrence_mat[index, frame2idx[frame]] = 1

        torch.save(co_occurrence_mat,
                   '/h/19/jadeleiyu/frame_extension/data/gsn/co_occurrence_mat_{}s.pt'.format(decade))
        query_noun_eval_df.to_csv('/h/19/jadeleiyu/frame_extension/data/gsn/noun_eval_df_{}s.csv'.format(decade))
        frame_eval_df.to_csv('/h/19/jadeleiyu/frame_extension/data/gsn/frame_eval_df_{}s.csv'.format(decade))
