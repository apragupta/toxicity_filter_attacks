from typing import List

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