#!/usr/bin/env python

import itertools
import logging
import sys

import math
import os

import numpy as np
import tensorflow as tf

import loader
from loader import augment_with_pretrained, calculate_global_maxes
from loader import update_tag_scheme, prepare_dataset
from loader import word_mapping, char_mapping, tag_mapping
# from model import Model
from model_tensorflow import Model
from utils import models_path, evaluate, eval_script, eval_temp
from utils import read_args, form_parameters_dict

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

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path, overwrite_mappings=opts.overwrite_mappings)
print "Model location: %s" % model.model_path

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

global_max_sentence_length, global_max_char_length = \
    calculate_global_maxes(max_sentence_lengths, max_word_lengths)

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
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_buckets, train_stats, train_unique_words = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id,
    global_max_sentence_length, global_max_char_length,
    lower
)
dev_buckets, dev_stats, dev_unique_words = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id,
    global_max_sentence_length, global_max_char_length,
    lower
)
test_buckets, test_stats, test_unique_words = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id,
    global_max_sentence_length, global_max_char_length,
    lower
)

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
model.save_mappings(id_to_word, id_to_char, id_to_tag)

batch_size = opts.batch_size

# Build the model
cost, train_step, _, _, _, _, _, enqueue_op, placeholders = model.build(max_sentence_length_scalar=global_max_sentence_length,
                                             max_word_length_scalar=global_max_char_length,
                                             batch_size_scalar=batch_size,
                                             **parameters)

# config = tf.ConfigProto(
#     device_count = {'GPU': 0})

config = None

sess = tf.Session(config=config)

model.sess = sess
model.saver = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=6, max_to_keep=200)

sess.run(tf.global_variables_initializer())

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    # model.reload()
    ckpt = tf.train.get_checkpoint_state(model.model_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print "Reloaded %s" % ckpt.model_checkpoint_path

### At this point, the training data is encoded in our format.

#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
n_epochs = opts.maximum_epochs  # number of epochs over the training set
freq_eval = int(len(train_stats)/5)  # evaluate on dev every freq_eval steps
best_dev = -np.inf
best_test = -np.inf
count = 0

for epoch in xrange(n_epochs):
    epoch_costs = []
    print "Starting epoch %i..." % epoch
    # training
    # form a batch first
    ## choose a bucket

    import threading
    from loader import _load_and_enqueue

    permuted_bucket_ids = np.random.permutation(range(len(train_buckets)))

    for bucket_id in list(permuted_bucket_ids):

    # bucket_id = np.random.random_integers(0, len(train_bins)-1)
        bucket_data = train_buckets[bucket_id][0]
        bucket_maxes = train_buckets[bucket_id][1]

        n_batches = int(math.ceil(float(bucket_data['sentence_lengths'].shape[0]) / batch_size))

        def load_and_enqueue():
            _load_and_enqueue(sess, bucket_data, n_batches, batch_size, placeholders,
                                                       enqueue_op,
                                                      train=True)

        t = threading.Thread(target=load_and_enqueue)
        t.start()

        print "n_batches: %d" % n_batches
        print "bucket_id: %d" % bucket_id

        for batch_idx in range(n_batches):
            count += batch_size

            cost_value, _ = sess.run([cost, train_step])
            epoch_costs.append(cost_value)
            if count % 50 == 0 and count != 0:
                sys.stdout.write("%f " % np.mean(epoch_costs[-50:]))
                sys.stdout.flush()
    model.save(epoch, best_performances=[best_dev, best_test], epoch_costs=epoch_costs)
    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
    print "Model dir: %s" % model.model_path
    t.join()