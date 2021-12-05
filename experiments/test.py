"""
Test abusive language classifier
"""

import tensorflow as tf
import os

import experiments.params as param
from data_utils import data_helpers, eval_helpers
from model.abuse_classifier import AbuseClassifier

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("pos_vocab_size", 26, "Vocab size of POS tags")
tf.flags.DEFINE_integer("pos_embedding_dim", 25, "Dimensionality of pos tag embedding (default: 20)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("attention_lambda", 0, "Supervised attention lambda (default: 0.05)")
tf.flags.DEFINE_string("attention_loss_type", 'encoded', "loss function of attention")
tf.flags.DEFINE_float("l2_reg_lambda", 0.02, "L2 regularizaion lambda (default: 0.05)")
tf.flags.DEFINE_integer("hidden_size", 300, "Dimensionality of RNN cell (default: 300)")
tf.flags.DEFINE_integer("pos_hidden_size", 25, "Dimensionality of POS-RNN cell")
tf.flags.DEFINE_integer("attention_size", 20, "Dimensionality of attention scheme (default: 50)")
tf.flags.DEFINE_boolean("use_pos_flag", True, "use the sequence of POS tags")
# Training parameters -- evaluate_every should be 100
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500000, "Save model after this many steps (default: 100)")
#tf.flags.DEFINE_float("train_ratio", 1.0, "Ratio of training data")
# Misc Parameters
tf.flags.DEFINE_string("checkpoint", '', "model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
"""
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
"""

def scoreUtil(init_embed, x_dev, pos_dev, y_dev, length_dev, pos_length_dev, model_path):
    with tf.Graph().as_default():
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        model = AbuseClassifier(
          max_sequence_length=param.max_sent_len,
          num_classes=2,
          pos_vocab_size = FLAGS.pos_vocab_size,
          init_embed=init_embed,
          hidden_size=FLAGS.hidden_size,
          attention_size=FLAGS.attention_size,
          keep_prob=FLAGS.dropout_keep_prob,
          attention_lambda = FLAGS.attention_lambda,
          attention_loss_type = FLAGS.attention_loss_type,
          l2_reg_lambda=0.1,
          use_pos_flag = FLAGS.use_pos_flag)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        saver = tf.train.Saver(tf.all_variables())
        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, model_path)

        dev_scores = []
        pos = 0
        gap = 50
        while (pos < len(x_dev)):
          x_batch = x_dev[pos:pos+gap]
          pos_batch = pos_dev[pos:pos+gap]
          y_batch = y_dev[pos:pos+gap]
          length_batch = length_dev[pos:pos+gap]
          pos_length_batch = pos_length_dev[pos:pos+gap]
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


def scoreComments(model_path, data_type="test"):
    """
    Score comments with saved model
    """
    vocabulary, pos_vocabulary, init_embed = data_helpers.loadVocabEmb()
    print("pos vocab size: {}".format(len(pos_vocabulary)))
    x_test, length_test, _, pos_test, pos_length_test, y_test = data_helpers.loadData(data_type)
    test_scores = scoreUtil(init_embed, x_test, pos_test, y_test, length_test, pos_length_test, model_path)
    gold_scores = [s[0] for s in y_test]
    return gold_scores, test_scores


if __name__=="__main__":
    # locate checkpoint
    if FLAGS.checkpoint == "":
        out_dir = os.path.abspath(os.path.join(os.path.pardir, "model"))
        print("Writing to {}\n".format(out_dir))
    else:
        out_dir = FLAGS.checkpoint
    if (FLAGS.attention_lambda == 0.0):
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model_noatt_checkpoints"))
    else:
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model_att="+FLAGS.attention_loss_type+"_checkpoints"))
    model_path = os.path.join(checkpoint_dir, "best_model")
                              
    # evaluate on train data
    train_gold_scores, train_pred_scores = scoreComments(model_path, data_type="train")
    # evaluate on test data
    test_gold_scores, test_pred_scores = scoreComments(model_path, data_type="test")
    
    # roc auc
    eval_helpers.evalROC(test_gold_scores, test_pred_scores)
    # pr auc
    eval_helpers.evalPR(test_gold_scores, test_pred_scores)
    # f1 score
    eval_helpers.evalFscore(train_gold_scores, train_pred_scores,
                            test_gold_scores, test_pred_scores)   

 
