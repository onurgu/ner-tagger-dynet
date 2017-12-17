#!/usr/bin/env python

import itertools
import logging
import sys
import time

from functools import partial

import math
import os

import numpy as np

import loader
from loader import augment_with_pretrained, calculate_global_maxes
from loader import update_tag_scheme, prepare_dataset
from loader import word_mapping, char_mapping, tag_mapping, morpho_tag_mapping

from model import MainTaggerModel
from utils import models_path, evaluate, eval_script, eval_temp
from utils import read_args, form_parameters_dict

from dynetsaver import DynetSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Read parameters from command line
opts = read_args()

# Parse parameters
parameters = form_parameters_dict(opts)

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['t_s'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if parameters['train_with_yuret']:
    parameters['test_with_yuret'] = 1

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# TODO: Move this to a better configurational structure
eval_logs_dir = os.path.join(eval_temp, "eval_logs")

if opts.model_path:
    model = MainTaggerModel(parameters=parameters,
                            models_path=models_path,
                            model_path=opts.model_path,
                            overwrite_mappings=opts.overwrite_mappings)
else:
    # Initialize model
    model = MainTaggerModel(parameters=parameters, models_path=models_path, overwrite_mappings=opts.overwrite_mappings)
print "MainTaggerModel location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['t_s']

max_sentence_lengths = {}
max_word_lengths = {}

# Load sentences
train_sentences, max_sentence_lengths['train'], max_word_lengths['train'] = loader.load_sentences(opts.train, lower, zeros)
dev_sentences, max_sentence_lengths['dev'], max_word_lengths['dev'] = loader.load_sentences(opts.dev, lower, zeros)
test_sentences, max_sentence_lengths['test'], max_word_lengths['test'] = loader.load_sentences(opts.test, lower, zeros)

if parameters['test_with_yuret'] or parameters['train_with_yuret']:
    # train.merge and test.merge
    yuret_train_sentences, max_sentence_lengths['yuret_train'], max_word_lengths['yuret_train'] = \
        loader.load_sentences(opts.yuret_train, lower, zeros)
    yuret_test_sentences, max_sentence_lengths['yuret_test'], max_word_lengths['yuret_test'] = \
        loader.load_sentences(opts.yuret_test, lower, zeros)
    update_tag_scheme(yuret_train_sentences, tag_scheme)
    update_tag_scheme(yuret_test_sentences, tag_scheme)
else:
    yuret_train_sentences = []
    yuret_test_sentences = []

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = \
    char_mapping(train_sentences + dev_sentences + test_sentences + yuret_train_sentences + yuret_test_sentences)
dico_tags, tag_to_id, id_to_tag = \
    tag_mapping(train_sentences + dev_sentences + test_sentences + yuret_train_sentences + yuret_test_sentences)
if parameters['mt_d'] > 0:
    dico_morpho_tags, morpho_tag_to_id, id_to_morpho_tag = \
        morpho_tag_mapping(train_sentences + dev_sentences + test_sentences + yuret_train_sentences + yuret_test_sentences,
                           morpho_tag_type=parameters['mt_t'],
                           morpho_tag_column_index=parameters['mt_ci'],
                           joint_learning=True)
else:
    id_to_morpho_tag = {}
    morpho_tag_to_id = {}

if opts.overwrite_mappings:
    print 'Saving the mappings to disk...'
    model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

model.reload_mappings()


# Index data
train_buckets, train_stats, train_unique_words, train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
)
dev_buckets, dev_stats, dev_unique_words, dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
)
test_buckets, test_stats, test_unique_words, test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
)

if parameters['test_with_yuret'] or parameters['train_with_yuret']:
    # yuret train and test datasets
    yuret_train_buckets, yuret_train_stats, yuret_train_unique_words, yuret_train_data = prepare_dataset(
        yuret_train_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
        lower, parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
    )
    yuret_test_buckets, yuret_test_stats, yuret_test_unique_words, yuret_test_data = prepare_dataset(
        yuret_test_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
        lower, parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
    )
else:
    yuret_train_buckets = []
    yuret_test_buckets = []

    yuret_train_data = []
    yuret_test_data = []

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_stats), len(dev_stats), len(test_stats))

print "%i / %i / %i words in train / dev / test." % (
    sum([x[0] for x in train_stats]), sum([x[0] for x in dev_stats]), sum([x[0] for x in test_stats]))

print "%i / %i / %i longest sentences in train / dev / test." % (
    max([x[0] for x in train_stats]), max([x[0] for x in dev_stats]), max([x[0] for x in test_stats]))

print "%i / %i / %i shortest sentences in train / dev / test." % (
    min([x[0] for x in train_stats]), min([x[0] for x in dev_stats]), min([x[0] for x in test_stats]))

for i, label in [[2, 'char']]:
    print "%i / %i / %i total %s in train / dev / test." % (
        sum([sum(x[i]) for x in train_stats]), sum([sum(x[i]) for x in dev_stats]), sum([sum(x[i]) for x in test_stats]),
        label)

    print "%i / %i / %i max. %s lengths in train / dev / test." % (
        max([max(x[i]) for x in train_stats]), max([max(x[i]) for x in dev_stats]), max([max(x[i]) for x in test_stats]),
        label)

    print "%i / %i / %i min. %s lengths in train / dev / test." % (
        min([min(x[i]) for x in train_stats]), min([min(x[i]) for x in dev_stats]), min([min(x[i]) for x in test_stats]),
        label)

print "Max. sentence lengths: %s" % max_sentence_lengths
print "Max. char lengths: %s" % max_word_lengths

for label, bucket_stats, n_unique_words in [['train', train_stats, train_unique_words],
                                            ['dev', dev_stats, dev_unique_words],
                                            ['test', test_stats, test_unique_words]]:

    int32_items = len(train_stats) * (max_sentence_lengths[label] * ( 5 + max_word_lengths[label] ) + 1)
    float32_items = n_unique_words * parameters['word_dim']
    total_size = int32_items + float32_items
    # TODO: fix this with byte sizes
    logging.info("Input ids size of the %s dataset is %d" % (label, int32_items))
    logging.info("Word embeddings (unique: %d) size of the %s dataset is %d" % (n_unique_words, label, float32_items))
    logging.info("Total size of the %s dataset is %d" % (label, total_size))

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

batch_size = opts.batch_size

# Build the model
model.build(**parameters)

model.saver = DynetSaver(model.model, model.model_path)

# Reload previous model values
if opts.reload or opts.model_path:
    print 'Reloading previous model...'
    # model.reload()
    model_checkpoint_path = model.saver.get_newest_ckpt_directory()
    if model_checkpoint_path:
        # Restores from checkpoint
        model.saver.restore(model_checkpoint_path)
        print "Reloaded %s" % model_checkpoint_path

### At this point, the training data is encoded in our format.

from eval import eval_with_specific_model

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = opts.maximum_epochs  # number of epochs over the training set
freq_eval = int(len(train_stats)/5)  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf

if model.parameters['active_models'] in [1, 2, 3]:
    best_morph_dev = -np.inf
    best_morph_test = -np.inf

count = 0

model.trainer.set_clip_threshold(5.0)

def get_loss_for_a_batch(batch_data,
                         loss_function=partial(model.get_loss, gungor_data=True),
                         label="G"):

    loss_value = update_loss(batch_data, loss_function)

    return loss_value


def yield_bucket_data(train_buckets):
    permuted_bucket_ids = np.random.permutation(range(len(train_buckets)))

    for bucket_id in list(permuted_bucket_ids):

        bucket_data = train_buckets[bucket_id]

        yield bucket_id, bucket_data


def yield_random_batches(data, batch_size=opts.batch_size):
    shuffled_data = data[np.random.permutation(range(len(data)))]

    index = 0
    while index < len(shuffled_data):
        yield shuffled_data[index:(index+batch_size)]
        index += batch_size


def yield_random_batches_from_bucket_data(train_buckets):

    for bucket_id, bucket_data in yield_bucket_data(train_buckets):

        it_batches = yield_random_batches(bucket_data)
        while True:
            try:
                yield bucket_id, it_batches.next()
            except StopIteration as e:
                print e


def update_loss(sentences_in_the_batch, loss_function):

    loss = loss_function(sentences_in_the_batch)
    loss.backward()
    model.trainer.update()
    if loss.value() / batch_size >= (10000000000.0 - 1):
        logging.error("BEEP")

    return loss.value()

for epoch in range(n_epochs):
    start_time = time.time()
    epoch_costs = []
    print "Starting epoch %i..." % epoch

    count = 0
    yuret_count = 0

    while True:
        try:
            if opts.use_buckets:
                bucket_id, batch_data = yield_random_batches_from_bucket_data(train_buckets)
                print "bucket_id: %d, len(batch_data): %d" % (bucket_id, len(batch_data))
            else:
                _, batch_data = yield_random_batches(train_data)

            epoch_costs += get_loss_for_a_batch(batch_data)
            print ""

            if model.parameters["train_with_yuret"]:
                bucket_id, batch_data = yield_random_batches_from_bucket_data(yuret_train_buckets)
                epoch_costs += get_loss_for_a_batch(batch_data,
                                     loss_function=partial(model.get_loss, gungor_data=False),
                                     label="Y")
                print ""

            count += len(data)

            model.trainer.status()

            if count % 50 == 0 and count != 0:
                sys.stdout.write("%s%f " % (label, np.mean(epoch_costs[-50:])))
                sys.stdout.flush()
                if np.mean(losses_of_this_bucket[-50:]) > 100:
                    logging.error("BEEP")
        except StopIteration as e:
            print e
    print ""

    buckets_to_be_tested = [("dev", dev_buckets),
                            ("test", test_buckets)]
    if model.parameters['test_with_yuret']:
        buckets_to_be_tested.append(("yuret", yuret_test_buckets))

    f_scores, morph_accuracies = eval_with_specific_model(model, epoch, buckets_to_be_tested,
                                        model.parameters['integration_mode'],
                                        model.parameters['active_models'],
                                        id_to_tag, batch_size,
                                        eval_logs_dir,
                                        tag_scheme
                                        )
    if model.parameters['active_models'] in [0, 2, 3]:
        if best_dev < f_scores["dev"]:
            print("NER Epoch: %d New best dev score => best_dev, best_test: %lf %lf" % (epoch + 1,
                                                                                               f_scores["dev"],
                                                                                               f_scores["test"]))
            best_dev = f_scores["dev"]
            best_test = f_scores["test"]
            model.save(epoch)
            model.save_best_performances_and_costs(epoch,
                                                   best_performances=[f_scores["dev"], f_scores["test"]],
                                                   epoch_costs=epoch_costs)
        else:
            print("NER Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf" % (epoch + 1,
                                                                                                       best_dev,
                                                                                                       best_test))

    if model.parameters['active_models'] in [1, 2, 3]:
        if best_morph_dev < morph_accuracies["dev"]:
            print("MORPH Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                  (epoch, morph_accuracies["dev"], morph_accuracies["test"]))
            best_morph_dev = morph_accuracies["dev"]
            best_morph_test = morph_accuracies["test"]
            if parameters['test_with_yuret']:
                best_morph_yuret = morph_accuracies["yuret"]
                print("YURET Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                      (epoch, 0.0, morph_accuracies["yuret"]))
                # we do not save in this case, just reporting
        else:
            print("MORPH Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                  % (epoch, best_morph_dev, best_morph_test))
            if parameters['test_with_yuret']:
                print("YURET Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                      % (epoch, 0.0, best_morph_yuret))

    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
    print "MainTaggerModel dir: %s" % model.model_path
    print "Training took %lf seconds for this epoch" % (time.time()-start_time)