import sys

import tensorflow as tf
import numpy as np
import os
import pickle

from data_utils.data_helpers import genFeatures, loadVocabEmb
from data_utils.tag_data_helpers import genPOSFeatures
from experiments import params
from model.abuse_classifier import AbuseClassifier

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("pos_vocab_size", 26, "Vocab size of POS tags")
tf.flags.DEFINE_integer("pos_embedding_dim", 25, "Dimensionality of pos tag embedding (default: 20)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("attention_lambda", 0, "Supervised attention lambda (default: 0.05)")
tf.flags.DEFINE_string("attention_loss_type", 'encoded', "loss function of attention")
tf.flags.DEFINE_float("l2_reg_lambda", 0.02, "L2 regularization lambda (default: 0.05)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of RNN cell (default: 300)")
tf.flags.DEFINE_integer("pos_hidden_size", 25, "Dimensionality of POS-RNN cell")
tf.flags.DEFINE_integer("attention_size", 20, "Dimensionality of attention scheme (default: 50)")
tf.flags.DEFINE_boolean("use_pos_flag", True, "use the sequence of POS tags")
# Training parameters -- evaluate_every should be 100
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500000, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_float("train_ratio", 1.0, "Ratio of training data")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS


def load_vocab():
    vocabulary, pos_vocabulary, init_embed = loadVocabEmb()
    return vocabulary, pos_vocabulary, init_embed


def load_data(dump_folder_path, data_type, verbose=True):
    assert data_type in ["train", "test"]
    with open(os.path.join(dump_folder_path, "vocab.pkl"), "rb") as handle:
        vocabulary = pickle.load(handle)
    with open(os.path.join(dump_folder_path, "pos_vocab.pkl"), "rb") as handle:
        pos_vocabulary = pickle.load(handle)
    with open(os.path.join(dump_folder_path, data_type + "_comm.data"), "rb") as handle:
        sentences, labels = pickle.load(handle)
    with open(os.path.join(dump_folder_path, data_type + "_comm_pos.data"), "rb") as handle:
        pos_sentences = pickle.load(handle)
    with open(os.path.join(dump_folder_path, data_type + "_attention.data"), "rb") as handle:
        attention = pickle.load(handle)
    # generate features & labels
    x, length, attention = genFeatures(sentences, attention, params.max_sent_len, vocabulary)
    pos, pos_length = genPOSFeatures(pos_sentences, params.max_sent_len, pos_vocabulary)
    y = np.array(labels)
    if verbose:
        print("load {} data, input sent size: {}, input POS size: {}, label size: {}".format(
            data_type, np.array(x).shape, np.array(pos).shape, np.array(y).shape))
    x_test, length_test, attention_test, pos_test, pos_length_test, y_test = x, length, attention, pos, pos_length, y
    return x_test, length_test, attention_test, pos_test, pos_length_test, y_test


def get_predictions(model_path, data_path):
    vocabulary, pos_vocabulary, init_embed = load_vocab()
    x_test, length_test, attention_test, pos_test, pos_length_test, y_test = load_data(data_path, "test", verbose=False)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AbuseClassifier(
                max_sequence_length=params.max_sent_len,
                num_classes=2,
                pos_vocab_size=FLAGS.pos_vocab_size,
                init_embed=init_embed,
                hidden_size=FLAGS.hidden_size,
                attention_size=FLAGS.attention_size,
                keep_prob=FLAGS.dropout_keep_prob,
                attention_lambda=FLAGS.attention_lambda,
                attention_loss_type=FLAGS.attention_loss_type,
                l2_reg_lambda=0.1,
                use_pos_flag=FLAGS.use_pos_flag)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            saver = tf.train.Saver(tf.all_variables())
            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, model_path)

            dev_scores = []
            pos = 0
            gap = 50
            while pos < len(x_test):
                x_batch = x_test[pos:pos + gap]
                pos_batch = pos_test[pos:pos + gap]
                y_batch = y_test[pos:pos + gap]
                length_batch = length_test[pos:pos + gap]
                pos_length_batch = pos_length_test[pos:pos + gap]
                pos += gap
                # score sentences
                feed_dict = {
                    model.input_word: x_batch,
                    model.input_pos: pos_batch,
                    model.input_y: y_batch,
                    model.sequence_length: length_batch,
                    model.dropout_keep_prob: 1.0
                }
                step, scores = sess.run([global_step, model.prob], feed_dict)
                dev_scores = dev_scores + list([s[0] for s in scores])
    return dev_scores


if __name__ == '__main__':

    model_save_folder_name = (sys.argv[1] if len(sys.argv) > 1 else False) or "model_noatt_checkpoints"

    print(f"Running model from {model_save_folder_name}")

    model_folder_path = '../model'
    checkpoint_dir = os.path.abspath(os.path.join(model_folder_path, model_save_folder_name))
    model_path = os.path.join(checkpoint_dir, "best_model")

    data_path = '../dump'
    print(f"Using data from {data_path}")

    predictions = get_predictions(model_path, data_path)

    print("Done")
    print("-" * 100)
    print(f"Ran model from {model_save_folder_name}, using data from {data_path}")

    print("Predictions:")
    print(predictions)

    print("Classification:")
    y_pred = list(map(lambda p: 1 if p > 0.5 else 0, predictions))
    print(y_pred)
