import os, sys
import re
import numpy as np

import dynet
from dynet import Model, BiRNNBuilder, LSTMBuilder, CoupledLSTMBuilder

import codecs
import cPickle

import logging

from crf import CRF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import get_name, create_a_model_subpath, get_model_subpath, \
    add_a_model_path_to_the_model_paths_database


class MainTaggerModel(object):
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

        self.model = Model()

        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # MainTaggerModel location
            # MainTaggerModel location
            available_model_subpath, model_path_id = create_a_model_subpath(models_path)
            # model_path = os.path.join(models_path, available_model_subpath)
            add_a_model_path_to_the_model_paths_database(models_path, available_model_subpath,
                                                         get_name(parameters))
            self.model_path = available_model_subpath
            # model_path = os.path.join(models_path, self.name)
            # self.model_path = model_path
            self.parameters_path = os.path.join(self.model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(self.model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f)
        else:
            # TODO: handle this part of reloading a saved model..
            assert parameters is None and models_path is None
            # MainTaggerModel location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = cPickle.load(f)
            self.reload_mappings()
        self.components = {}

    def save_mappings(self, id_to_word, id_to_char, id_to_tag, id_to_morpho_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        self.id_to_morpho_tag = id_to_morpho_tag

        if os.path.exists(self.mappings_path) and not self.overwrite_mappings:
            print "Aborting. A previous mappings file exists. You should explicitly state to overwrite the mappings file"
            sys.exit(1)
        else:
            with open(self.mappings_path, 'wb') as f:
                mappings = {
                    'id_to_word': self.id_to_word,
                    'id_to_char': self.id_to_char,
                    'id_to_tag': self.id_to_tag,
                    'id_to_morpho_tag': self.id_to_morpho_tag,
                }
                cPickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']
        self.id_to_morpho_tag = mappings['id_to_morpho_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self, epoch=None, best_performances=[], epoch_costs=[]):
        """
        Write components values to disk.
        """
        model_dir_path = self.model_path

        # for name, param in self.components.items():
        #     param_path = os.path.join(path, "%s.mat" % name)
        #     if hasattr(param, 'params'):
        #         param_values = {p.name: p.get_value() for p in param.params}
        #     else:
        #         param_values = {name: param.get_value()}
        #     scipy.io.savemat(param_path, param_values)

        self.saver.save(epoch=epoch, n_bests=self.n_bests)

        self.save_best_performances_and_costs(epoch, best_performances, epoch_costs)

    def save_best_performances_and_costs(self, epoch, best_performances, epoch_costs):

        path = self.model_path
        model_ckpt_filename = ("model-epoch-%08d" % epoch) if epoch is not None else (
        "best-models-%08d" % self.n_bests)
        if len(best_performances) > 0:
            best_performances_path = os.path.join(path,
                                                  "%s-%s.txt" % (
                                                  model_ckpt_filename, "best_performances"))
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

        self.saver.restore(os.path.join(path, "model.ckpt"),
                           epoch=epoch, n_bests=self.n_bests)

    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              ch_b,
              mt_d,
              mt_t,
              word_dim,
              word_lstm_dim,
              w_b,
              lr_method,
              pre_emb,
              crf,
              cap_dim,
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
        n_morpho_tags = len(self.id_to_morpho_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 17

        # Final input (all word features)
        word_representation_dim = 0

        def get_scale(shape):
            return np.sqrt(6/np.sum(list(shape)))

        #
        # Word inputs
        #
        if word_dim:
            # Initialize with pretrained embeddings
            scale = get_scale((n_words, word_dim))
            new_weights = scale * np.random.uniform(-1.0, 1.0, (n_words, word_dim))
            # new_weights = np.zeros([n_words, word_dim], dtype='float32')
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
            self.word_embeddings = self.model.add_lookup_parameters((n_words, word_dim),
                                                                    init=dynet.NumpyInitializer(
                                                                        new_weights),
                                                                    name="wordembeddings")

            self.tanh_layer_W = self.model.add_parameters((word_lstm_dim, 2 * word_lstm_dim))
            self.tanh_layer_b = self.model.add_parameters((word_lstm_dim))

            self.last_layer_W = self.model.add_parameters((n_tags, word_lstm_dim))
            self.last_layer_b = self.model.add_parameters((n_tags))

        def create_bilstm_layer(label, input_dim, lstm_dim, bilstm=True):
            if bilstm:
                builder = BiRNNBuilder(1, input_dim, lstm_dim, self.model, CoupledLSTMBuilder)
            else:
                builder = CoupledLSTMBuilder(1, input_dim, lstm_dim, self.model)

            return builder

        self.representation_creation_model_and_input = dict()
        # Chars inputs
        #
        if char_dim:
            self.char_embeddings = self.model.add_lookup_parameters((n_chars, char_dim),
                                                                    name="charembeddings")

            self.char_lstm_layer = create_bilstm_layer("char",
                                                       char_dim,
                                                       (2 if ch_b else 1) * char_lstm_dim,
                                                       bilstm=True if ch_b else False)

            self.representation_creation_model_and_input['char'] = (self.char_embeddings,
                                                                    self.char_lstm_layer,
                                                                    'char_for_ids')

            word_representation_dim += (2 if ch_b else 1) * char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            word_representation_dim += cap_dim
            self.cap_embeddings = self.model.add_lookup_parameters((n_cap, cap_dim),
                                                                   name="capembeddings")

        if mt_d > 0:

            self.morpho_tag_embeddings = self.model.add_lookup_parameters((n_morpho_tags, mt_d),
                                                                              name="charembeddings")

            self.morpho_tag_lstm_layer_for_golden_morpho_analyzes = \
                create_bilstm_layer("morpho_tag_lstm_layer_for_golden_morpho_analyzes",
                                    mt_d,
                                    2 * mt_d,
                                    bilstm=True)

            self.representation_creation_model_and_input['morpho_tag'] = (self.morpho_tag_embeddings,
                                                                    self.morpho_tag_lstm_layer_for_golden_morpho_analyzes,
                                                                    'morpho_tag_ids')

            word_representation_dim += 2 * mt_d

        # LSTM for words
        self.sentence_level_bilstm_layer = \
            create_bilstm_layer("sentence_level",
                                word_representation_dim,
                                2 * word_lstm_dim,
                                bilstm=True if w_b else False)

        self.crf_module = CRF(self.model, self.id_to_tag)



        # Training
        def process_hyperparameter_definition(x):
            tokens = x.split("@")
            subtokens = tokens[0].split("_")
            if len(subtokens) > 1 and subtokens[-1] == "float":
                return ["_".join(subtokens[:-1]), float(tokens[1])]
            else:
                return tokens
        _tokens = lr_method.split("-")
        opt_update_algorithm = _tokens[0]
        opt_hyperparameters = [process_hyperparameter_definition(x) for x in _tokens[1:]]
        opt_update_algorithms = {'sgd': dynet.SimpleSGDTrainer,
                                 'adam': dynet.AdamTrainer,
                                 'adadelta': dynet.AdadeltaTrainer,
                                 'adagrad': dynet.AdagradTrainer,
                                 'momentum': dynet.MomentumSGDTrainer,
                                 'rmsprop': dynet.RMSPropTrainer}

        self.trainer = opt_update_algorithms[opt_update_algorithm](self.model,
                                                                   **{name: value for name, value in opt_hyperparameters})

        # self.trainer = dynet.SimpleSGDTrainer(self.model, learning_rate=0.01)

        return self

    def get_char_representations(self, sentence):
        return self.get_representations(sentence, 'char')

    def get_morpho_tag_representations(self, sentence):
        return self.get_representations(sentence, 'morpho_tag')

    def get_representations(self, sentence, label):
        # initial_state = self.char_lstm_layer.initial_state()

        embeddings = self.representation_creation_model_and_input[label][0]
        lstm_layer = self.representation_creation_model_and_input[label][1]
        input_label = self.representation_creation_model_and_input[label][2]

        char_embeddings = [[embeddings[char_id] for char_id in word]
                           for sentence_pos, word in enumerate(sentence[input_label])]

        char_representations = []
        for sentence_pos, char_embeddings_for_word in enumerate(char_embeddings):
            # print char_embeddings_for_word
            try:
                char_representations.append(
                    lstm_layer.transduce(char_embeddings_for_word)[-1])
            except IndexError as e:
                print sentence
                print char_embeddings_for_word
                print e
        return char_representations

    def get_sentence_level_bilstm_outputs(self, combined_word_representations):

        context_representations = \
            self.sentence_level_bilstm_layer.transduce(combined_word_representations)

        context_representations = [dynet.tanh(dynet.affine_transform([self.tanh_layer_b.expr(),
                                                                      self.tanh_layer_W.expr(),
                                                                      context])) \
                                   for context in context_representations]
        return context_representations

    def get_loss(self, sentences_in_the_batch):
        # immediate_compute=True, check_validity=True
        dynet.renew_cg()
        loss_array = []
        for sentence in sentences_in_the_batch:
            """
            data.append({
                'str_words': str_words,
                'word_ids': words,
                'char_for_ids': chars,
                'char_lengths': [len(char) for char in chars],
                'cap_ids': caps,
                'tag_ids': tags,
                # if mt_d > 0: 'morpho_tag_ids': morpho_tags,
                'sentence_lengths': len(s),
                'max_word_length_in_this_sample': max([len(x) for x in chars])
            })
            """

            tag_scores = self.calculate_tag_scores(sentence)

            loss = self.crf_module.neg_log_loss(tag_scores, sentence['tag_ids'])

            loss_array.append(loss)
        return dynet.esum(loss_array)

    def calculate_tag_scores(self, sentence):
        word_embedding_based_representations = \
            [self.word_embeddings[word_id] for word_id in sentence['word_ids']]
        char_representations = self.get_char_representations(sentence)
        cap_embedding_based_representations = \
            [self.cap_embeddings[cap_id] for cap_id in sentence['cap_ids']]
        if self.parameters['mt_d'] > 0:
            morph_tag_based_representations = self.get_morpho_tag_representations(sentence)
        else:
            morph_tag_based_representations = None
        combined_word_representations = [dynet.concatenate(list(zipped_reps)) for zipped_reps in
                                         zip(*filter(lambda x: x is not None,
                                                    [word_embedding_based_representations,
                                                     char_representations,
                                                     morph_tag_based_representations,
                                                     cap_embedding_based_representations]))]
        # print combined_word_representations
        # print self.parameters
        combined_word_representations = [dynet.dropout(x, p=self.parameters['dropout'])
                                         for x in combined_word_representations]
        context_representations = \
            self.get_sentence_level_bilstm_outputs(combined_word_representations)
        tag_scores = [dynet.affine_transform([self.last_layer_b.expr(),
                                              self.last_layer_W.expr(),
                                              context]) \
                      for context in context_representations]
        return tag_scores
