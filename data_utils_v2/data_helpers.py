"""
Data helpers for model training
"""
import numpy as np
import os
import pickle
import copy
import pandas as pd

# hyperparameter
max_sent_len = 100
unk = "<UNK>"
pad = "<PAD/>"
emb_dim = 300

def loadVocabEmb(dump_folder):
    # vocabulary & embedding
    with open(os.path.join(dump_folder, "vocab.pkl"), "rb") as handle:
        vocabulary = pickle.load(handle)
    with open(os.path.join(dump_folder, "pos_vocab.pkl"), "rb") as handle:
        pos_vocabulary = pickle.load(handle)
    with open(os.path.join(dump_folder, "norm_init_embed.pkl"), "rb") as handle:
        init_embed = pickle.load(handle)
    return vocabulary, pos_vocabulary, init_embed


def splitTrainData(data, train_ratio=0.8, verbose=True):
    # split train data into train & dev sets
    data_size = len(data[0])
    groups = len(data)
    train_size = int(data_size * train_ratio)
    train_inds = set(np.random.choice(range(data_size), size=train_size, replace=False))
    train_data = [[] for t in range(groups)]
    dev_data = [[] for t in range(groups)]
    for ind in range(data_size):
        if ind in train_inds:
            for t in range(groups):
                train_data[t].append(copy.deepcopy(data[t][ind]))
        else:
            for t in range(groups):
                dev_data[t].append(copy.deepcopy(data[t][ind]))
    if verbose:
        print("split into train ({} examples) and dev sets ({} examples)".format(len(train_data[0]), len(dev_data[0])))
    return train_data + dev_data


def loadData(dump_folder, data_folder, data_type, verbose=True, type="comments"):
    assert data_type in ["train", "test"]
    with open(os.path.join(dump_folder, "vocab.pkl"), "rb") as handle:
        vocabulary = pickle.load(handle)
    with open(os.path.join(dump_folder, "pos_vocab.pkl"), "rb") as handle:
        pos_vocabulary = pickle.load(handle)


    df_path = os.path.join(data_folder, data_type + f"_{type}_df")
    data_df = pd.read_pickle(df_path)
    sentences = data_df["tokenized"].to_list()
    if(type=="comments"):
        labels = data_df["merged_label"].to_list()
    else:
        labels = data_df["binarized_label"].to_list()

    pos_sentences = data_df["pos_tags"].to_list()
    attention = data_df["attention"].to_list()

    # generate features & labels
    x, length, attention = genFeatures(sentences, attention, max_sent_len, vocabulary)
    pos, pos_length = genPOSFeatures(pos_sentences, max_sent_len, pos_vocabulary)
    y = np.array(labels)
    if verbose:
        print("load {} data, input sent size: {}, input POS size: {}, label size: {}".format(
            data_type, np.array(x).shape, np.array(pos).shape, np.array(y).shape))
    return x, length, attention, pos, pos_length, y


def loadTrainData(dump_folder, data_folder, type="comments"):
    data = loadData(dump_folder, data_folder, data_type="train", verbose=True, type=type)
    return splitTrainData(data, train_ratio=0.8, verbose=True)


def loadTestData(dump_folder, data_folder, type="comments"):
    return loadData(dump_folder, data_folder, data_type="test", verbose=True, type=type)


def padSents(sentences, max_len, padding_word=pad):
    # length_list = np.array([len(sent) for sent in sentences])
    length_list = []
    padded_sentences = []
    for i in range(len(sentences)):

        sent = sentences[i][:max_len]

        num_padding = max_len - len(sent)
        new_sentence = sent + [padding_word] * num_padding
        length_list.append(len(new_sentence))
        padded_sentences.append(new_sentence)
    return padded_sentences, np.array(length_list)


def genFeatures(sent_list, attention_list, max_sent_len, vocabulary):
    # pad sentences

    padded_sent_list, length_list = padSents(sent_list, max_sent_len)
    padded_attention_list, _ = padSents(attention_list, max_sent_len, 0)
    print("padded sent:", np.array(padded_sent_list).shape)
    # generate features
    x = []
    for sent in padded_sent_list:
        sent_x = []
        for word in sent:
            try:
                sent_x.append(vocabulary[word])
            except:
                sent_x.append(vocabulary[unk])
                continue
        x.append(sent_x[:])
    x = np.array(x)
    # x = np.array([[vocabulary[word] for word in sent] for sent in padded_sent_list])
    padded_attention_list = np.array(padded_attention_list)
    print("feature shape:", np.array(x).shape)
    return x, length_list, padded_attention_list


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    print("Num Steps: ", num_batches_per_epoch*num_epochs)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

"""
POS tag helpers for model training
"""



def padPOSSents(pos_sentences, max_len, padding_pos="<POS/>"):
    length_list = []
    padded_pos_sentences = []
    for i in range(len(pos_sentences)):
        sent = pos_sentences[i][:max_len]
        num_padding = max_len - len(sent)
        new_sentence = sent + [padding_pos] * num_padding
        length_list.append(len(new_sentence))
        padded_pos_sentences.append(new_sentence[:])
    return padded_pos_sentences, np.array(length_list)


def cleanPOSSents(pos_sentences, pos_vocabulary, unk_pos=unk):
    # replace pos tags not in pos_vocabulary with unk
    for (sent_ind, pos_sent) in enumerate(pos_sentences):
        for (word_ind, word) in enumerate(pos_sent):
            if word not in pos_vocabulary:
                pos_sentences[sent_ind][word_ind] = unk_pos


def genPOSFeatures(pos_sentences, max_sent_len, pos_vocabulary, verbose=True):
    padded_pos_sentences, length_list = padPOSSents(pos_sentences, max_sent_len)
    cleanPOSSents(padded_pos_sentences, pos_vocabulary)
    x = np.array([[pos_vocabulary[word] for word in sent] for sent in padded_pos_sentences])
    if verbose:
        print("padded pos sentences:", np.array(padded_pos_sentences).shape)
        print("debug padded_pos_sentences:", padded_pos_sentences[0][:10])
        print("pos feature shape:", np.array(x).shape)
    return x, length_list