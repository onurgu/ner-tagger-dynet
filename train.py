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
# from model import MainTaggerModel
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

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# TODO: Move this to a better configurational structure
eval_logs_dir = os.path.join(eval_temp, "eval_logs")

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

# train.merge and test.merge
yuret_train_sentences, max_sentence_lengths['yuret_train'], max_word_lengths['yuret_train'] = \
    loader.load_sentences(opts.yuret_train, lower, zeros)
yuret_test_sentences, max_sentence_lengths['yuret_test'], max_word_lengths['yuret_test'] = \
    loader.load_sentences(opts.yuret_test, lower, zeros)
update_tag_scheme(yuret_train_sentences, tag_scheme)
update_tag_scheme(yuret_test_sentences, tag_scheme)

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
dico_morpho_tags, morpho_tag_to_id, id_to_morpho_tag = \
    morpho_tag_mapping(train_sentences + dev_sentences + test_sentences + yuret_train_sentences + yuret_test_sentences,
                       morpho_tag_type=parameters['mt_t'],
                       morpho_tag_column_index=parameters['mt_ci'],
                       joint_learning=True)

if opts.overwrite_mappings:
    print 'Saving the mappings to disk...'
    model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

model.reload_mappings()


# Index data
train_buckets, train_stats, train_unique_words = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_t'], parameters['mt_ci'],
)
dev_buckets, dev_stats, dev_unique_words = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_t'], parameters['mt_ci'],
)
test_buckets, test_stats, test_unique_words = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_t'], parameters['mt_ci'],
)

# yuret train and test datasets
yuret_train_buckets, yuret_train_stats, yuret_train_unique_words = prepare_dataset(
    yuret_train_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_t'], parameters['mt_ci'],
)
yuret_test_buckets, yuret_test_stats, yuret_test_unique_words = prepare_dataset(
    yuret_test_sentences, word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
    lower, parameters['mt_t'], parameters['mt_ci'],
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
model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

batch_size = opts.batch_size

# Build the model
model.build(**parameters)

model.saver = DynetSaver(model.model, model.model_path)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    # model.reload()
    ckpt = model.saver.get_checkpoint_state()
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        model.saver.restore(ckpt.model_checkpoint_path)
        print "Reloaded %s" % ckpt.model_checkpoint_path

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

if model.parameters['integration_mode'] > 0:
    best_morph_dev = -np.inf
    best_morph_test = -np.inf

count = 0

model.trainer.set_clip_threshold(5.0)

for epoch in xrange(n_epochs):
    start_time = time.time()
    epoch_costs = []
    print "Starting epoch %i..." % epoch

    permuted_bucket_ids = np.random.permutation(range(len(train_buckets)))

    def get_loss_for_bucket_data(bucket_id, bucket_data, count,
                                 loss_function=partial(model.get_loss, gungor_data=True),
                                 label="G"):
        n_batches = int(math.ceil(float(len(bucket_data)) / batch_size))

        print "bucket_id: %d, n_batches: %d" % (bucket_id, n_batches)

        losses_of_this_bucket = []

        for batch_idx in range(n_batches):
            count += batch_size

            sentences_in_the_batch = bucket_data[(batch_idx*batch_size):((batch_idx+1)*batch_size)]

            loss = loss_function(sentences_in_the_batch)
            loss.backward()
            model.trainer.update()
            if loss.value()/batch_size >= (10000000000.0 - 1):
                logging.error("BEEP")
            losses_of_this_bucket.append(loss.value()/batch_size)
            # epoch_costs.append(loss.value()/batch_size)
            if count % 50 == 0 and count != 0:
                sys.stdout.write("%s%f " % (label, np.mean(losses_of_this_bucket[-50:])))
                sys.stdout.flush()
                if np.mean(epoch_costs[-50:]) > 100:
                    logging.error("BEEP")

        return losses_of_this_bucket

    count = 0
    yuret_count = 0

    for bucket_id in list(permuted_bucket_ids):

        # train on gungor_data
        bucket_data = train_buckets[bucket_id]


        get_loss_for_bucket_data(bucket_id, bucket_data, count)
        print ""

        if model.parameters['train_with_yuret']:
            # train on yuret data
            yuret_bucket_data = yuret_train_buckets[bucket_id]

            get_loss_for_bucket_data(bucket_id, yuret_bucket_data, yuret_count,
                                     loss_function=partial(model.get_loss, gungor_data=False),
                                     label="Y")
            print ""
        model.trainer.status()

    print ""
    f_scores, morph_accuracies = eval_with_specific_model(model, epoch, [("dev", dev_buckets),
                                                                         ("test", test_buckets),
                                                                         ("yuret", yuret_test_buckets)],
                                        model.parameters['integration_mode'],
                                        id_to_tag, batch_size,
                                        eval_logs_dir,
                                        tag_scheme
                                        )
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

    if model.parameters['integration_mode'] > 0:
        if best_morph_dev < morph_accuracies["dev"]:
            print("MORPH Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                  (epoch, morph_accuracies["dev"], morph_accuracies["test"]))
            best_morph_dev = morph_accuracies["dev"]
            best_morph_test = morph_accuracies["test"]
            best_morph_yuret = morph_accuracies["yuret"]
            print("YURET Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                  (epoch, 0.0, morph_accuracies["yuret"]))
            # we do not save in this case, just reporting
        else:
            print("MORPH Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                  % (epoch, best_morph_dev, best_morph_test))
            print("YURET Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                  % (epoch, 0.0, best_morph_yuret))

    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
    print "MainTaggerModel dir: %s" % model.model_path
    print "Training took %lf seconds for this epoch" % (time.time()-start_time)