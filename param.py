import os

# path
data_folder = os.path.join(os.getcwd(), "data")
dump_folder = os.path.join(os.getcwd(), "dump_2")
classification_dataset = os.path.join(data_folder, "Anonymized_Sentences_Classified.csv")
categorization_dataset = os.path.join(data_folder, "Anonymized_Comments_Categorized.csv")

# hyperparameter
max_sent_len = 100
unk = "<UNK>"
pad = "<PAD/>"
emb_dim = 300

