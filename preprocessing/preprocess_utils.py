#this file has multiple utils streamlining the preprocessing of data from the raw form

#imports


import re
import pandas as pd
import preprocessor as p
import os
import itertools
from tqdm import tqdm
from twitter_pos_tagger import get_pos_of_file


max_sent_len = 100
unk = "<UNK>"
pad = "<PAD/>"
emb_dim = 300



#utils
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()$,!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def clean_raw_data(raw_df):
    print(f"length before filtering {len(raw_df)}")
    # keep track of list of row indices to drop
    to_drop = []
    unlabeled_cnt = 0
    empty_sent_cnt = 0
    empty_comment_cnt = 0
    for idx, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0]):
        comment_ind = row["Comment #"]
        sent_label = str(row["Abusive"])
        sent = row["Sentence"]

        # try to cast comment to int
        try:
            comment_ind = int(comment_ind)
        except:
            print("empty comment_ind", idx)
            empty_comment_cnt += 1
            to_drop.append(idx)
            continue

        # skip any empty sentences
        if (pd.isna(sent) or (sent.strip == "")):
            print("empty sentence", idx)
            empty_sent_cnt += 1
            to_drop.append(idx)
            continue
        if (sent_label.strip() == "No"):
            sent_label = 0
        elif (sent_label.strip() == "Yes"):
            sent_label = 1
        else:
            print("Missing label in comment: {} sentence {}".format(comment_ind, idx))
            unlabeled_cnt += 1
            to_drop.append(idx)
            continue

    print("# of unlabeld sents: {}".format(unlabeled_cnt))
    print("# of sents without comment: {}".format(empty_comment_cnt))
    print("# of empty sentences {} ".format(empty_sent_cnt))

    # create new filtered dataframe
    filtered_df = raw_df.drop(to_drop)
    print(f"length after filtering {len(filtered_df)}")
    return filtered_df

def add_binarized_label(clean_data):
    """
    Creates and adds binarized label out of "Abusive" column
    :param clean_data:
    :return:
    """
    clean_data['label'] = clean_data['Abusive'].apply(lambda row: 0 if row == "No" else 1)
    clean_data['binarized_label'] = clean_data['label'].apply(lambda row: [row, 1 - row])


def gen_comment_df(clean_data, verbose=True):
    # util to create a dataframe thats joined into comments with required columns
    # column called label must be added
    prev_comment_ind = clean_data.loc[0]["Comment #"]
    comm_sent_list = []
    comm_label_list = []
    tmp_sent_list = []
    tmp_label_list = []
    comment_idx_list = [prev_comment_ind]
    comment_range_list = []

    start_ind = 0

    for idx, row in tqdm(clean_data.iterrows(), total=clean_data.shape[0]):
        comment_ind = row["Comment #"]
        sent_label = row["label"]
        sent = row["Sentence"]

        sent = sent.strip()
        if (comment_ind == prev_comment_ind):
            tmp_sent_list.append(sent)
            tmp_label_list.append(sent_label)
        else:
            # store prev comment
            comm_sent_list.append(tmp_sent_list[:])
            comm_label_list.append(tmp_label_list[:])

            end_ind = idx - 1
            comment_range_list.append([start_ind, end_ind])

            # start is now the current index
            start_ind = idx

            # update tmp
            tmp_sent_list = [sent]
            tmp_label_list = [sent_label]
            prev_comment_ind = comment_ind
            comment_idx_list.append(comment_ind)

    # store the last comment
    end_ind = start_ind + len(tmp_sent_list)
    comment_range_list.append([start_ind, end_ind])
    comm_sent_list.append(tmp_sent_list[:])
    comm_label_list.append(tmp_label_list[:])
    # sanity check: print last two comments and labels
    if verbose:
        for i in range(1, 3):
            print("example comment:", comm_sent_list[-i], "labels:", comm_label_list[-i])
    assert (len(comm_sent_list) == len(comm_label_list)), "length of labels and comments don't match"
    comments_df = pd.DataFrame(columns=['Comment', 'labels', 'comment_idx', "sent_range"])

    comments_df['Comment'] = comm_sent_list
    comments_df['labels'] = comm_label_list
    comments_df['comment_idx'] = comment_idx_list
    comments_df['sent_range'] = comment_range_list

    comments_df["merged_comment"] = comments_df["Comment"].apply(lambda row: " ".join(row))
    comments_df["merged_label"] = comments_df["labels"].apply(lambda row: [max(row), 1 - max(row)])


    return comments_df

def tokenize_df(data_df,phrase_col):
    """
    Adds 'tokenized' column to data_df which is a tokenized version of the phrases in phrase_col
    :param data:
    :param col:
    :param phrase_col:
    :return:
    """
    data_df["tokenized"] = data_df[phrase_col].apply(lambda row: clean_str(p.tokenize(row)).split())


def write_to_corpus(tokenized_comment_df,corpus_path):
    # save comment level tokenized data to corpus: newline seperated
    # No need to run this since we have already created a corpus
    sent_list = tokenized_comment_df["tokenized"].to_list()
    corpus_file =  open(corpus_path, "w")
    corpus_file.write("\n".join([" ".join(sent) for sent in sent_list]))
    corpus_file.write("\n")

    #dont forget to close the corpus file!
    corpus_file.close()


def gen_pos_tags(data_df, out_folder_path, raw_col):
    ##adds a row containing POS tags to the given dataframe pased on the column containing row data
    comment_list = data_df[raw_col].to_list()
    pos_comment_list = []
    max_sent_num = 5000
    ind = 0

    tweet_pos_folder = os.path.join(out_folder_path, "for_pos")
    if not os.path.isdir(tweet_pos_folder):
        os.mkdir(tweet_pos_folder)

    for i in range(5):
        print(comment_list[i])
    print(f"generating {raw_col} POS TAGS")
    while (ind < len(comment_list)):
        print(f"gen {ind} tags")
        comm_list = comment_list[ind: ind + max_sent_num]

        # name of file to save newline seperated comments in
        for_pos_filename = f"{raw_col}_{ind}.txt"

        # path of file to save newline seperated comments in
        for_pos_filepath = os.path.join(tweet_pos_folder, for_pos_filename)

        # first, we write these tweets (newline seperated) into a text file
        with open(for_pos_filepath, 'w', encoding="utf-8") as f:
            for comm in comm_list:
                f.write(str(comm + '\n'))

        # then, we can run the pos tagger on that file and get the tags for each token
        pos_tags = get_pos_of_file(for_pos_filepath)

        # for each comment
        pos_list = [comm[1] for comm in pos_tags]
        pos_comment_list = pos_comment_list + pos_list

        ind += max_sent_num
    assert len(pos_comment_list) == len(data_df)
    data_df['pos_tags'] = pos_comment_list


def preprocess_save_raw_df(raw_input_file_path,out_folder_path):
    # creating paths
    if not os.path.isdir(out_folder_path):
        os.mkdir(out_folder_path)

    # csv folder to save preprocessed comment level data in
    out_comment_path = os.path.join(out_folder_path, 'comments_df')
    # csv folder to save preprocessed sentence level data in
    out_sentence_path = os.path.join(out_folder_path, 'sentence_df')


    data = pd.read_csv(raw_input_file_path)
    clean_data = clean_raw_data(data)
    clean_data['sentence_idx'] = clean_data.index
    add_binarized_label(clean_data)
    comments_df = gen_comment_df(clean_data)

    tokenize_df(clean_data,"Sentence")
    tokenize_df(comments_df, "merged_comment")

    # add attention vector to sentence data
    clean_data['attention'] = clean_data.apply(lambda row: [row['label']] * len(row['tokenized']), axis=1)

    # now add attention to comment level data simply by cross referencing the sentence range indices
    comments_df["attention"] = comments_df["sent_range"].apply(
        lambda row: list(itertools.chain.from_iterable(clean_data.iloc[row[0]:row[1]]['attention'].to_list())))

    gen_pos_tags(comments_df, out_folder_path, "merged_comment")
    gen_pos_tags(clean_data, out_folder_path, "Sentence")



    #save the processed csv
    comments_df.to_pickle(out_comment_path)
    clean_data.to_pickle(out_sentence_path)



