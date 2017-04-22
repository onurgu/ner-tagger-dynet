import os, sys
import re
import numpy as np

#import scipy.io

import codecs
import cPickle

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import get_name
# from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward
# from optimization import Optimization

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn, static_rnn

from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_rnn

class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None, overwrite_mappings=0):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """

        self.n_bests = 0
        self.overwrite_mappings = overwrite_mappings

        self.saver = None

        self.sess = None

        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                self.parameters = cPickle.dump(parameters, f)
        else:
            # TODO: handle this part of reloading a saved model..
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag

        if self.overwrite_mappings:
            with open(self.mappings_path, 'wb') as f:
                mappings = {
                    'id_to_word': self.id_to_word,
                    'id_to_char': self.id_to_char,
                    'id_to_tag': self.id_to_tag,
                }
                cPickle.dump(mappings, f)
        elif os.path.exists(self.mappings_path):
            print "Aborting. A previous mappings file exists. You should explicitly state to overwrite the mappings file"
            sys.exit(1)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self, epoch=-1, best_performances=[], epoch_costs=[]):
        """
        Write components values to disk.
        """
        path = self.model_path
        model_ckpt_filename = "model-epoch-%08d" % epoch if epoch != -1 else "best-models-%08d" % self.n_bests

        # for name, param in self.components.items():
        #     param_path = os.path.join(path, "%s.mat" % name)
        #     if hasattr(param, 'params'):
        #         param_values = {p.name: p.get_value() for p in param.params}
        #     else:
        #         param_values = {name: param.get_value()}
        #     scipy.io.savemat(param_path, param_values)

        assert self.sess is not None, "Session is not created yet, you cannot save."
        self.saver.save(self.sess,
                        os.path.join(path, model_ckpt_filename),
                        global_step=(epoch if epoch != -1 else self.n_bests))

        if len(best_performances) > 0:
            best_performances_path = os.path.join(path, "%s-%s.txt" % (model_ckpt_filename, "best_performances"))
            best_performances_f = open(best_performances_path, "w")
            best_performances_f.write(" ".join([str(b) for b in best_performances]) + "\n")
            best_performances_f.close()
        if len(epoch_costs) > 0:
            epoch_cost_path = os.path.join(path, "%s-%s.txt" % (model_ckpt_filename, "epoch_cost"))
            epoch_cost_f = open(epoch_cost_path, "w")
            epoch_cost_f.write(" ".join([str(e) for e in epoch_costs]) + "\n")
            epoch_cost_f.write(str(np.mean(epoch_costs)) + "\n")
            epoch_cost_f.close()

    def reload(self, epoch=-1):
        """
        Load components values from disk.
        """
        path = self.model_path
        if epoch != -1:
            path = os.path.join(path, "epoch-%08d" % epoch)
        else:
            path = os.path.join(path, "best-model-%08d" % self.n_bests)
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.restore(self.sess,
                           os.path.join(path, "model.ckpt"),
                           global_step=(epoch if epoch != -1 else self.n_bests))

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              ch_b,
              word_dim,
              word_lstm_dim,
              w_b,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
              max_sentence_length_scalar,
              max_word_length_scalar,
              batch_size_scalar,
              training=True,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 17

        sentence_level_sequence_shape = [max_sentence_length_scalar]
        word_level_character_sequence_shape = [max_sentence_length_scalar, max_word_length_scalar]

        is_train_feed = tf.placeholder(tf.bool, shape=(), name='is_train')

        max_sentence_length = tf.constant(max_sentence_length_scalar, shape=(), dtype=tf.int32)
        max_word_length = tf.constant(max_word_length_scalar, shape=(), dtype=tf.int32)

        placeholders = {}

        placeholders['is_train'] = is_train_feed

        with tf.name_scope("inputs"):
            word_ids_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar] + sentence_level_sequence_shape, name="word_ids_feed")
            sentence_lengths_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar], name="sentence_lengths_feed")
            char_for_ids_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar] + word_level_character_sequence_shape, name="char_for_ids_feed")
            char_lengths_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar] + sentence_level_sequence_shape, name="char_lengths_feed")
            tag_ids_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar] + sentence_level_sequence_shape, name="tag_ids_feed")
            placeholders['word_ids'] = word_ids_feed
            placeholders['sentence_lengths'] = sentence_lengths_feed
            placeholders['char_for_ids'] = char_for_ids_feed
            placeholders['char_lengths'] = char_lengths_feed
            placeholders['tag_ids'] = tag_ids_feed

            if cap_dim:
                cap_ids_feed = tf.placeholder(tf.int32, shape=[batch_size_scalar] + sentence_level_sequence_shape, name="cap_ids_feed")
                placeholders['cap_ids'] = cap_ids_feed

        # Final input (all word features)
        word_representation_dim = 0
        inputs = []

        q = tf.FIFOQueue(10000, 6 * [tf.int32] + [tf.bool], shapes=[[batch_size_scalar] + sentence_level_sequence_shape,
                                                        [batch_size_scalar],
                                                        [batch_size_scalar] + word_level_character_sequence_shape,
                                                        [batch_size_scalar] + sentence_level_sequence_shape,
                                                        [batch_size_scalar] + sentence_level_sequence_shape,
                                                        [batch_size_scalar] + sentence_level_sequence_shape,
                                                        []])

        enqueue_op = q.enqueue([word_ids_feed,
                                sentence_lengths_feed,
                                char_for_ids_feed,
                                char_lengths_feed,
                                tag_ids_feed,
                                cap_ids_feed,
                                is_train_feed])

        word_ids, sentence_lengths, char_for_ids, char_lengths, \
        tag_ids, cap_ids, is_train = \
            q.dequeue()

        #
        # Word inputs
        #
        if word_dim:
            # Initialize with pretrained embeddings
            new_weights = np.zeros([n_words, word_dim], dtype='float32')
            if pre_emb and training:
                print 'Loading pretrained embeddings from %s...' % pre_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    raw_word = self.id_to_word[i]
                    if raw_word != "<UNK>":
                        # word = raw_word.split(" ")[1]
                        word = raw_word
                    else:
                        word = raw_word
                    # print word
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1

                print 'Loaded %i pretrained embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      )
            word_representation_dim += word_dim
            word_embeddings = tf.get_variable("word_embeddings", initializer=new_weights)
            words = tf.nn.embedding_lookup(word_embeddings, word_ids)
            # (batch_size, max_sentence_length, word_dim)
            inputs.append(words)

        from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn

        def create_bilstm_layer(label, lstm_dim, sequences_3d_tensor, sequence_lengths, bilstm=True):
            with tf.variable_scope("%s_lstm" % label):
                with tf.variable_scope("%s_cells_fw" % label):
                    cell_fw_char = LSTMCell(lstm_dim)
                if bilstm:
                    with tf.variable_scope("%s_cells_bw" % label):
                        cell_bw_char = LSTMCell(lstm_dim)

                if bilstm:
                    bilstm_outputs, final_state_tuples_fw_and_bw = \
                        bidirectional_dynamic_rnn(cell_fw=cell_fw_char,
                                                  cell_bw=cell_bw_char,
                                                  inputs=sequences_3d_tensor,
                                                  time_major=True,
                                                  sequence_length=sequence_lengths,
                                                  dtype=tf.float32)
                else:
                    bilstm_outputs, final_state_tuples_fw_and_bw = \
                        dynamic_rnn(cell=cell_fw_char,
                                                  inputs=sequences_3d_tensor,
                                                  time_major=True,
                                                  sequence_length=sequence_lengths,
                                                  dtype=tf.float32)
            return final_state_tuples_fw_and_bw, bilstm_outputs

        #
        # Chars inputs
        #
        if char_dim:
            char_embeddings = tf.get_variable("char_embeddings", shape=[n_chars, char_dim])

            chars_forward = tf.nn.embedding_lookup(char_embeddings, char_for_ids)
            # (batch_size, max_sentence_length, max_word_length, char_dim)

            # char_lengths
            # (batch_size, max_sentence_length)

            character_sequences_3d_tensor = tf.transpose(
                tf.reshape(chars_forward,
                           [batch_size_scalar * max_sentence_length_scalar, max_word_length_scalar,
                            char_dim]),
                [1, 0, 2])
            if ch_b:
                char_level_bilstm_state_tuples, _ = create_bilstm_layer("char",
                                                          char_lstm_dim,
                                                          character_sequences_3d_tensor,
                                                          tf.reshape(char_lengths, [batch_size_scalar*max_sentence_length_scalar]))
                char_for_outputs = tf.reshape(
                    tf.concat([char_level_bilstm_state_tuples[0][1],
                               char_level_bilstm_state_tuples[1][1]], axis=1),
                    [batch_size_scalar, max_sentence_length_scalar, 2 * char_lstm_dim])
                # (batch_size, max_sentence_length, 2*char_lstm_dim)
            else:
                char_level_bilstm_state_tuples, _ = create_bilstm_layer("char",
                                                          char_lstm_dim,
                                                          character_sequences_3d_tensor,
                                                          tf.reshape(char_lengths, [batch_size_scalar*max_sentence_length_scalar]),
                                                          bilstm=False)
                char_for_outputs = tf.reshape(char_level_bilstm_state_tuples, [batch_size_scalar, max_sentence_length_scalar, char_lstm_dim])
            # print char_for_outputs
            inputs.append(char_for_outputs)
            word_representation_dim += char_lstm_dim
            if ch_b:
            #     inputs.append(char_rev_output)
                word_representation_dim += char_lstm_dim


        #
        # Capitalization feature
        #
        if cap_dim:
            word_representation_dim += cap_dim
            cap_embeddings = tf.get_variable("cap_embeddings",
                                                    shape=[n_cap, cap_dim])
            cap_embeddings = tf.nn.embedding_lookup(cap_embeddings, cap_ids)
            inputs.append(cap_embeddings)

        # Prepare final input
        if len(inputs) != 1:
            word_representations = tf.concat(inputs, axis=2, name='word_representations_concat') # 3rd axis as the first axis is for batch id

        #
        # Dropout on final input
        #
        if dropout:
            dropout_output = tf.nn.dropout(word_representations, keep_prob=dropout, name="dropout")

            input_train = dropout_output
            input_test = (1 - dropout) * word_representations
            word_representations = tf.cond(tf.equal(is_train, True), lambda: input_train, lambda: input_test, name='word_representations')

        word_representations_3d_tensor = tf.transpose(
            word_representations,
            [1, 0, 2])

        # LSTM for words
        if w_b:
            word_level_bilstm_state_tuples, word_level_bilstm_outputs_fw_bw = \
                create_bilstm_layer("word_level",
                                    word_representation_dim,
                                    word_representations_3d_tensor,
                                    sentence_lengths)

            context_representations = tf.transpose(
                tf.concat([word_level_bilstm_outputs_fw_bw[0], word_level_bilstm_outputs_fw_bw[1]], axis=2),
                [1, 0, 2])

            tanh_dense_output = tf.layers.dense(context_representations,
                                                word_lstm_dim,
                                                activation=tf.nn.tanh,
                                                name='tanh_dense_output')
            # no need to transpose again. tanh_dense_output = tf.transpose(tanh_dense_output, [1, 0, 2])
        else:
            word_level_bilstm_state_tuples, word_level_bilstm_outputs_fw = \
                create_bilstm_layer("word_level",
                                    word_representation_dim,
                                    word_representations_3d_tensor,
                                    sentence_lengths,
                                    bilstm=False)

            context_representations = tf.transpose(
                word_level_bilstm_outputs_fw,
                [1, 0, 2])

            tanh_dense_output = context_representations

        tag_scores = tf.layers.dense(tanh_dense_output,
                                     n_tags,
                                     activation=(None if crf else tf.nn.softmax),
                                     name='tag_scores')
        if not crf:
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=tag_scores, labels=tag_ids)
        else:
            from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode, crf_sequence_score, crf_binary_score
            with tf.variable_scope("trainable_params"):
                crf_transition_params = tf.get_variable("crf_transition_params", shape=(n_tags, n_tags), dtype=tf.float32)
            cost, _ = crf_log_likelihood(tag_scores, tag_ids,
                                      sequence_lengths=sentence_lengths,
                                      transition_params=crf_transition_params)

        # for training run train_step setting is_train to True in feed_dict
        with tf.name_scope('train'):
            # TODO: change according to lr_method
            train_step = tf.train.AdamOptimizer(1e-4).minimize(-cost)

        # for evaluation and prediction use tag_scores setting is_train to False in feed_dict
        # (make the decoding outside tensorflow)

        return cost, train_step, tag_scores, tag_ids, word_ids, crf_transition_params, sentence_lengths, enqueue_op, placeholders
