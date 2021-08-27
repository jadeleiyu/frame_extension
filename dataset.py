import random
import torch
from torch.utils.data import Dataset


class GSNFrameDataset(Dataset):
    def __init__(self, learning_df, noun2idx, n_q, n_s, learning_type='train'):
        self.learning_df = learning_df
        self.noun2idx = noun2idx
        self.learning_type = learning_type
        self.n_q = n_q
        self.n_s = n_s

    def __len__(self):
        if self.learning_type == 'test_noun':
            return len(self.learning_df['query noun'])
        else:
            return len(self.learning_df['frame'])

    def __getitem__(self, i):
        if self.learning_type == 'train':
            frame_id = self.learning_df['frame id'][i]
            nouns = self.learning_df['established nouns'][i]
            noun_counts = self.learning_df['established noun counts'][i]
            nouns_idx = list(range(len(nouns)))
            random.shuffle(nouns_idx)
            support_nouns_row_idx = nouns_idx[:self.n_s]
            support_nouns_idx = torch.tensor([self.noun2idx[nouns[k]] for k in support_nouns_row_idx])
            support_noun_counts = [noun_counts[k] for k in support_nouns_row_idx]
            support_nouns_total_count = sum(support_noun_counts)
            # query_nouns_row_idx = nouns_idx[:self.n_q]
            # query_nouns_idx = torch.tensor([self.noun2idx[nouns[k]] for k in query_nouns_row_idx])

            query_nouns = random.choices(self.learning_df['novel nouns train'][i], k=self.n_q)
            query_nouns_idx = torch.tensor([self.noun2idx[query_noun] for query_noun in query_nouns])

            return frame_id, query_nouns_idx, support_nouns_idx, support_nouns_total_count

        elif self.learning_type == 'val':
            frame_id = self.learning_df['frame id'][i]
            support_nouns = self.learning_df['established nouns'][i]
            support_noun_counts = self.learning_df['established noun counts'][i]
            support_nouns_row_idx = list(range(len(support_nouns)))
            random.shuffle(support_nouns_row_idx)
            sampled_support_nouns_row_idx = support_nouns_row_idx[:self.n_s]
            sampled_support_nouns = [support_nouns[k] for k in sampled_support_nouns_row_idx]
            sampled_support_noun_counts = [support_noun_counts[k] for k in sampled_support_nouns_row_idx]
            sampled_support_nouns_total_count = sum(sampled_support_noun_counts)
            sampled_support_nouns_idx = torch.tensor(
                [self.noun2idx[support_noun] for support_noun in sampled_support_nouns])

            query_nouns = random.choices(self.learning_df['novel nouns evaluation'][i], k=self.n_q)
            query_nouns_idx = torch.tensor([self.noun2idx[query_noun] for query_noun in query_nouns])
            return frame_id, query_nouns_idx, sampled_support_nouns_idx, sampled_support_nouns_total_count

        elif self.learning_type == 'test_noun':
            query_noun = self.learning_df['query noun'][i]
            return self.noun2idx[query_noun]
        else:
            support_nouns = self.learning_df['support nouns'][i]
            support_noun_counts = self.learning_df['support noun counts'][i]
            support_nouns_total_count = sum(support_noun_counts)
            novel_nouns = self.learning_df['ground truth novel nouns'][i]
            novel_nouns_query_idx = [self.noun2idx[novel_noun] for novel_noun in novel_nouns]
            novel_nouns_query_idx += [-1] * (self.n_q - len(novel_nouns_query_idx))
            novel_nouns_query_idx = torch.tensor(novel_nouns_query_idx)
            support_nouns_idx = torch.tensor([self.noun2idx[noun] for noun in support_nouns])
            return support_nouns_idx, support_nouns_total_count, novel_nouns_query_idx


class VisDataset(Dataset):
    def __init__(self, preprocessed_img_means):
        super(VisDataset, self).__init__()
        self.preprocessed_img_means = preprocessed_img_means

    def __getitem__(self, noun_id):
        return self.preprocessed_img_means[noun_id]
