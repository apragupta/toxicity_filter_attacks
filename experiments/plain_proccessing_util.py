import os
from typing import List
from preprocessing.twitter_pos_tagger import get_pos_of_file
import numpy as np


def read_raw_comment_txt(file_path, encoding="utf8", verbose=True, print_n=2):
    """
    Load the data - We will load plain text sentences, not the training data.
    The format is a .csv file with 1 sentence per line. The first column is the comment id, the second is the sentence
    text. These comments are preprocessed into lists of sentences each

    :param file_path:
    :param encoding:
    :param verbose:
    :param print_n:
    :return:
    """
    with open(file_path, "rt", encoding=encoding) as csv_file:
        prev_comment_id = 1
        comm_sent_list: List[List[str]] = []
        tmp_sent_list = []
        for row in csv_file.readlines():
            comment_id, sent = row.split(",")
            # Skip if not comment id
            try:
                comment_id = int(comment_id)
            except:
                print("empty comment_ind", row)
                continue
            sent = sent.strip()
            # skip empty sentence
            if sent == "":
                continue

            if comment_id == prev_comment_id:
                tmp_sent_list.append(sent)
            else:
                # store prev comment
                comm_sent_list.append(tmp_sent_list[:])

                # update tmp
                tmp_sent_list = [sent]
                prev_comment_id = comment_id

        # store the last comment
        comm_sent_list.append(tmp_sent_list[:])

        # sanity check: print two random comments and labels
        if verbose:
            for _ in range(print_n):
                n = np.random.randint(0, len(comm_sent_list))
                print(f"example comment (#{n}: {comm_sent_list[n]}")
        return comm_sent_list


def gen_pos_tags(comment_list, dump_folder_name, out_pos_name):
    tweet_pos_folder = os.path.join(dump_folder_name, "for_pos")
    pos_comment_list = []
    max_sent_num = 5000
    ind = 0

    while ind < len(comment_list):
        comm_list = comment_list[ind: ind + max_sent_num]

        # path of file to save newline seperated comments in
        for_pos_filepath = os.path.join(tweet_pos_folder, out_pos_name)

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

    return pos_comment_list
