from typing import List
from experiments.plain_proccessing_util import read_raw_comment_txt
from preprocessing.preproc_classification_data import clean_str, mapCommSent
import preprocessor as preprocessor

# Steps:
# 0. Load the data - We will load plain text sentences, not the training data.
# The format is a .csv file with 1 sentence per line. The first column is the comment id, the second is the sentence
# text. These comments are preprocessed into lists of sentences each


raw_comments_path = "comments.csv"
comments_list: List[List[str]] = read_raw_comment_txt(raw_comments_path)

# 1. Preprocess
# 1.a. Remove any new line characters
comments_list = [list(map(lambda sent: sent.replace("\n", ""), comment)) for comment in comments_list]
# 1.b. Remove any empty sentences
comments_list = [[sent for sent in comment if sent != ""] for comment in comments_list]
# 1.c. Join sentences with a space
map_dict = mapCommSent(comments_list)
merged_comments_list = [" ".join(sent_list) for sent_list in comments_list]
# 1.d. Tokenize the sentences
tokenized_merged_comments_list = [preprocessor.tokenize(sent) for sent in merged_comments_list]
# 1.e. Some other preprocessing? "clean_str"
cleaned_merged_comments_list = [clean_str(sent) for sent in merged_comments_list]

# 1.f. Get POS data

# 2. Load the model
# 3. Predict
# 4. Output
