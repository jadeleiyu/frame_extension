import os
import pickle
import random

import numpy as np
import torch
from PIL import Image
from nltk.corpus import wordnet
from torchvision import transforms
from tqdm import tqdm


def get_img_paths(imagenet_dir='/hal9000/datasets/imagenet/image/'):
    try:
        image_paths = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_image_paths.p', 'rb'))
        wnids = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_wnids.p', 'rb'))
    except FileNotFoundError:
        image_paths = []
        wnids = []
        for s in tqdm(os.listdir(imagenet_dir)):
            if os.path.isdir(os.path.join(imagenet_dir, s)):
                wnid_dir = os.path.join(imagenet_dir, s)
                wnids.append(s)
                for f in os.listdir(wnid_dir):
                    if f.endswith('.JPEG'):
                        image_paths.append(f)
        wnids = list(set(wnids))
        pickle.dump(image_paths, open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_image_paths.p', 'wb'))
        pickle.dump(wnids, open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_wnids.p', 'wb'))
    return image_paths, wnids


def get_wnid2img_path_idx(image_paths):
    try:
        wnid2img_path_idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_wnid2img_path_idx.p', 'rb'))
    except FileNotFoundError:
        wnid2img_path_idx = {}
        for i in range(len(image_paths)):
            wnid = image_paths[i].split('_')[0]
            if wnid not in wnid2img_path_idx:
                wnid2img_path_idx[wnid] = [i]
            else:
                wnid2img_path_idx[wnid].append(i)
        pickle.dump(wnid2img_path_idx, open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_wnid2img_path_idx.p', 'wb'))
    return wnid2img_path_idx


def get_word2wnids(wnids):
    try:
        word2wnids = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_word2wnids.p', 'rb'))
    except FileNotFoundError:
        word2wnids = {}
        for wnid in wnids:
            ss = wordnet.synset_from_pos_and_offset(pos='n', offset=int(wnid[1:]))
            for lemma in ss.lemmas():
                word = lemma.name()
                if word not in word2wnids:
                    word2wnids[word] = [wnid]
                else:
                    word2wnids[word].append(wnid)
        pickle.dump(word2wnids, open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_word2wnids.p', 'wb'))
    return word2wnids


def get_word2img_idx(word2wnids, wnid2img_path_idx):
    try:
        word2img_idx = pickle.load(open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_word2img_idx.p', 'rb'))
    except FileNotFoundError:
        valid_img_words = [word for word in word2wnids.keys()]
        word2img_idx = {}
        for word in valid_img_words:
            img_path_idx = []
            for wnid in word2wnids[word]:
                img_path_idx += wnid2img_path_idx[wnid]
            word2img_idx[word] = img_path_idx
        pickle.dump(word2img_idx, open('/h/19/jadeleiyu/frame_extension/data/img/imagenet_word2img_idx.p', 'wb'))
    return word2img_idx


def compute_visual_representations(noun2idx, image_paths, word2img_idx, img_sample_size=64):
    img_dir = '/hal9000/datasets/imagenet/image/'
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    idx2noun = {noun2idx[noun]: noun for noun in noun2idx.keys()}
    x_means = []
    num_exps = 0
    for i in tqdm(range(len(idx2noun)), position=0, leave=True):
        noun = idx2noun[i]
        paths = [image_paths[j] for j in word2img_idx[noun]]
        sampled_paths = random.sample(paths, img_sample_size)
        xs = []
        for image_path in sampled_paths:
            try:
                wnid = image_path.split('_')[0]
                img_fn = os.path.join(img_dir, wnid, image_path)
                raw_img = Image.open(img_fn)
                if len(np.array(raw_img).shape) == 3:
                    x = preprocess(raw_img)
                else:
                    x = torch.rand(3, 224, 224)
            except Exception as e:
                num_exps += 1
                x = torch.rand(3, 224, 224)
            xs.append(x)
        x_mean = torch.mean(torch.stack(xs), dim=0)
        x_means.append(x_mean)

    x_means = torch.stack(x_means)
    torch.save(x_means, '/h/19/jadeleiyu/frame_extension/data/img/noun_image_means.pt')
    print("{} out of {} support words have invalid imagenet representations".format(num_exps, len(idx2noun)))
#     return x_means
