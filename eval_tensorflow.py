"""Evaluation

"""
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

from datetime import datetime
import codecs
import itertools
import math
import logging
import os
import sys
import time



import numpy as np
import tensorflow as tf

from utils import read_args, form_parameters_dict, models_path, eval_script, eval_temp, iobes_iob

import loader
from loader import calculate_global_maxes, update_tag_scheme, \
    word_mapping, augment_with_pretrained, char_mapping, tag_mapping, prepare_dataset

from model_tensorflow import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")

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
event_logs_path = os.path.join(eval_temp, "eval_logs")
# if not os.path.exists(event_logs_path):
#     os.makedirs(event_logs_path)

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
train_sentences, max_sentence_lengths['train'], max_word_lengths['train'] =\
    loader.load_sentences(opts.train, lower, zeros)
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


if opts.overwrite_mappings:
    print 'Saving the mappings to disk...'
    model.save_mappings(id_to_word, id_to_char, id_to_tag)

model.reload_mappings()

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

for label, bin_stats, n_unique_words in [['train', train_stats, train_unique_words],
                                         ['dev', dev_stats, dev_unique_words],
                                         ['test', test_stats, test_unique_words]]:

    int32_items = len(train_stats) * (max_sentence_lengths[label] * ( 5 + max_word_lengths[label] ) + 1)
    float32_items = n_unique_words * parameters['word_dim']
    total_size = int32_items + float32_items
    logging.info("Input ids size of the %s dataset is %d" % (label, int32_items))
    logging.info("Word embeddings (unique: %d) size of the %s dataset is %d" % (n_unique_words, label, float32_items))
    logging.info("Total size of the %s dataset is %d" % (label, total_size))

from tensorflow.contrib.crf import viterbi_decode

batch_size = 5

# Build the model
cost, train_step, tag_scores, tag_ids, word_ids, \
crf_transition_params, sentence_lengths, enqueue_op, placeholders = model.build(max_sentence_length_scalar=global_max_sentence_length,
                                             max_word_length_scalar=global_max_char_length,
                                             batch_size_scalar=batch_size,
                                             **parameters)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', event_logs_path,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', model.model_path,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once():
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      summary_op: Summary op.
    """
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:

        model.sess = sess
        model.saver = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=6,
                                     max_to_keep=200)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print "Evaluating %s" % ckpt.model_checkpoint_path
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[-1])
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        import threading
        from loader import _load_and_enqueue

        for dataset_label, dataset_buckets in [["dev", dev_buckets], ["test", test_buckets]]:

            print "Starting to evaluate %s dataset" % dataset_label
            predictions = []
            n_tags = len(id_to_tag)
            count = np.zeros((n_tags, n_tags), dtype=np.int32)

            # permuted_bucket_ids = np.random.permutation(range(len(dataset_buckets)))

            for bucket_id in range(len(dataset_buckets)):

                # bucket_id = np.random.random_integers(0, len(train_bins)-1)
                bucket_data = dataset_buckets[bucket_id][0]
                bucket_maxes = dataset_buckets[bucket_id][1]

                n_batches = int(math.ceil(float(bucket_data['sentence_lengths'].shape[0]) / batch_size))

                print "dataset_label: %s" % dataset_label
                print ("n_batches: %d" % n_batches)
                print ("bucket_id: %d" % bucket_id)

                def load_and_enqueue():
                    _load_and_enqueue(sess, bucket_data, n_batches, batch_size, placeholders,
                                      enqueue_op,
                                      train=False)

                t = threading.Thread(target=load_and_enqueue)
                t.start()

                for batch_idx in range(n_batches):
                    # print("batch_idx: %d" % batch_idx)
                    sys.stdout.write(". ")
                    sys.stdout.flush()

                    tag_scores_value, tag_ids_value, word_ids_value, sentence_lengths_value = \
                        sess.run([tag_scores, tag_ids, word_ids, sentence_lengths])

                    for sentence_idx, one_sentence in enumerate(tag_scores_value):
                        sentence_length = sentence_lengths_value[sentence_idx]
                        # print sentence_idx
                        # print one_sentence[:sentence_length]
                        decoded_tags, _ = viterbi_decode(one_sentence[:sentence_length], crf_transition_params.eval())

                        p_tags = [id_to_tag[p_tag] for p_tag in decoded_tags]
                        r_tags = [id_to_tag[p_tag] for p_tag in tag_ids_value[sentence_idx, :sentence_length]]

                        if parameters['t_s'] == 'iobes':
                            p_tags = iobes_iob(p_tags)
                            r_tags = iobes_iob(r_tags)
                        for i, (word_id, y_pred, y_real) in enumerate(zip(word_ids_value[sentence_idx, :sentence_length], decoded_tags, tag_ids_value[sentence_idx, :sentence_length])):
                            new_line = " ".join([id_to_word[word_id]] + [r_tags[i], p_tags[i]])
                            predictions.append(new_line)
                            count[y_real, y_pred] += 1
                        predictions.append("")

                t.join()

            # print predictions

            # Write predictions to disk and run CoNLL script externally
            eval_id = np.random.randint(1000000, 2000000)
            output_path = os.path.join(FLAGS.eval_dir, "eval.%i.output" % eval_id)
            scores_path = os.path.join(FLAGS.eval_dir, "eval.%i.scores" % eval_id)
            with codecs.open(output_path, 'w', 'utf8') as f:
                f.write("\n".join(predictions))

            print "Evaluating the %s dataset with conlleval script" % dataset_label
            os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

            # CoNLL evaluation results
            eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
            for line in eval_lines:
                print line


def evaluate():
  """Eval CIFAR-10 for a number of steps.""" # with tf.Graph().as_default() as g:

  # Get images and labels for CIFAR-10.

  eval_data = FLAGS.eval_data == 'test'

  # # Build the summary operation based on the TF collection of Summaries.
  # summary_op = tf.summary.merge_all()
  #
  # summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

  while True:
    eval_once()
    if FLAGS.run_once:
      break
    print "Sleeping for %d" % FLAGS.eval_interval_secs
    time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
