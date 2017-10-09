"""Evaluation

"""
from __future__ import absolute_import
from __future__ import division

from collections import defaultdict as dd
import itertools
import logging
import math
import sys
import time

import subprocess

import codecs
import numpy as np

import os

import dynet

import loader
from loader import calculate_global_maxes, update_tag_scheme, \
    word_mapping, augment_with_pretrained, char_mapping, tag_mapping, prepare_dataset
from model import MainTaggerModel
from utils import read_args, form_parameters_dict, models_path, eval_script, eval_temp, iobes_iob
from dynetsaver import DynetSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")



def eval_once(model, dev_buckets, test_buckets, model_dir_path, integration_mode,
              run_for_all_checkpoints=False,
              *args):
    """Run Eval once.

    Args:
      saver: DynetSaver.
      summary_writer: Summary writer.
      summary_op: Summary op.
    """

    model.saver = DynetSaver(model.model, model_dir_path)
    ckpt = model.saver.get_checkpoint_state()
    if ckpt:
        if run_for_all_checkpoints:
            for model_checkpoint_path in ckpt.all_model_checkpoint_paths:
                eval_for_a_checkpoint(model.saver, model, model_checkpoint_path, dev_buckets, test_buckets,
                                      integration_mode,
                                      *args)
        else:
            eval_for_a_checkpoint(model.saver, model, ckpt.model_checkpoint_path, dev_buckets, test_buckets,
                                  integration_mode, *args)


def eval_for_a_checkpoint(saver, model, model_checkpoint_path, dev_buckets, test_buckets, integration_mode, *args):
    if model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(model_checkpoint_path)
        print "Evaluating %s" % model_checkpoint_path
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        epoch = int(os.path.basename(model_checkpoint_path).split('-')[-1])
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        return

    return eval_with_specific_model(model.model, epoch, dev_buckets, test_buckets, integration_mode, *args)


def eval_with_specific_model(model, epoch, buckets_list, integration_mode,
                             *args): # FLAGS.eval_dir
    # type: (MainTaggerModel, int, list, object, object) -> object
    id_to_tag, batch_size, eval_dir, tag_scheme = args

    f_scores = {}
    dataset_labels = ["dev", "test", "yuret"]

    total_correct_disambs = {dataset_label: 0 for dataset_label in dataset_labels}
    total_disamb_targets = {dataset_label: 0 for dataset_label in dataset_labels}
    if integration_mode > 0:
        detailed_correct_disambs = {dataset_label: dd(int) for dataset_label in dataset_labels}
        detailed_total_target_disambs = {dataset_label: dd(int) for dataset_label in dataset_labels}

    for dataset_label, dataset_buckets in buckets_list:

        print "Starting to evaluate %s dataset" % dataset_label
        predictions = []
        n_tags = len(id_to_tag)
        count = np.zeros((n_tags, n_tags), dtype=np.int32)

        # permuted_bucket_ids = np.random.permutation(range(len(dataset_buckets)))

        for bucket_id in range(len(dataset_buckets)):

            # bucket_id = np.random.random_integers(0, len(train_bins)-1)
            bucket_data_dict = dataset_buckets[bucket_id]

            n_batches = int(math.ceil(float(len(bucket_data_dict)) / batch_size))

            print "dataset_label: %s" % dataset_label
            print ("n_batches: %d" % n_batches)
            print ("bucket_id: %d" % bucket_id)

            for batch_idx in range(n_batches):
                # print("batch_idx: %d" % batch_idx)
                sys.stdout.write(". ")
                sys.stdout.flush()

                sentences_in_the_batch = bucket_data_dict[
                                         (batch_idx * batch_size):((batch_idx + 1) * batch_size)]

                for sentence in sentences_in_the_batch:
                    dynet.renew_cg()

                    sentence_length = len(sentence['word_ids'])

                    if integration_mode > 0:
                        selected_morph_analyzes, decoded_tags = model.predict(sentence)
                    elif integration_mode == 0:
                        decoded_tags = model.predict(sentence)

                    p_tags = [id_to_tag[p_tag] for p_tag in decoded_tags]
                    r_tags = [id_to_tag[p_tag] for p_tag in sentence['tag_ids']]

                    if integration_mode > 0:
                        n_correct_morph_disambs = \
                            sum([x == y for x, y, z in zip(selected_morph_analyzes,
                                                        sentence['golden_morph_analysis_indices'],
                                                        sentence['morpho_analyzes_tags']) if len(z) > 1])
                        total_correct_disambs[dataset_label] += n_correct_morph_disambs
                        total_disamb_targets[dataset_label] += sum([1 for el in sentence['morpho_analyzes_tags'] if len(el) > 1])
                        for key, value in [(len(el), x == y) for el, x, y in zip(sentence['morpho_analyzes_tags'],
                                                               selected_morph_analyzes,
                                                               sentence['golden_morph_analysis_indices'])]:
                            if value:
                                detailed_correct_disambs[dataset_label][key] += 1
                            detailed_total_target_disambs[dataset_label][key] += 1
                        # total_possible_analyzes += sum([len(el) for el in sentence['morpho_analyzes_tags'] if len(el) > 1])

                    if tag_scheme == 'iobes':
                        p_tags = iobes_iob(p_tags)
                        r_tags = iobes_iob(r_tags)
                    for i, (word_id, y_pred, y_real) in enumerate(
                            zip(sentence['word_ids'], decoded_tags,
                                sentence['tag_ids'])):
                        new_line = " ".join([sentence['str_words'][i]] + [r_tags[i], p_tags[i]])
                        predictions.append(new_line)
                        count[y_real, y_pred] += 1
                    predictions.append("")
            print ""

        # Write predictions to disk and run CoNLL script externally
        eval_id = np.random.randint(1000000, 2000000)
        output_path = os.path.join(eval_dir,
                                   "%s.eval.%i.epoch-%04d.output" % (dataset_label, eval_id, epoch))
        scores_path = os.path.join(eval_dir,
                                   "%s.eval.%i.epoch-%04d.scores" % (dataset_label, eval_id, epoch))
        with codecs.open(output_path, 'w', 'utf8') as f:
            f.write("\n".join(predictions))

        print "Evaluating the %s dataset with conlleval script" % dataset_label
        command_string = "%s < %s > %s" % (eval_script, output_path, scores_path)
        print command_string
        # os.system(command_string)
        # sys.exit(0)
        with codecs.open(output_path, "r", encoding="utf-8") as output_path_f:
            eval_lines = [x.rstrip() for x in subprocess.check_output([eval_script], stdin=output_path_f).split("\n")]

            # CoNLL evaluation results
            # eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
            for line in eval_lines:
                print line
            f_scores[dataset_label] = float(eval_lines[1].split(" ")[-1])

        if integration_mode > 0:
            for n_possible_analyzes in map(int, detailed_correct_disambs[dataset_label].keys()):
                print "%s %d %d/%d" % (dataset_label,
                                       n_possible_analyzes,
                                       detailed_correct_disambs[dataset_label][n_possible_analyzes],
                                       detailed_total_target_disambs[dataset_label][n_possible_analyzes])
    if integration_mode == 0:
        return f_scores, {}
    else:
        return f_scores, {dataset_label: total_correct_disambs[dataset_label]/float(total_disamb_targets[dataset_label]) for dataset_label in dataset_labels}

def evaluate(model, dev_buckets, test_buckets, opts, *args):
  """Eval CIFAR-10 for a number of steps.""" # with tf.Graph().as_default() as g:

  while True:
    eval_once(model, dev_buckets, test_buckets, model.model_path,
              opts.integration_mode,
              run_for_all_checkpoints=bool(opts.run_for_all_checkpoints),
              *args)
    print "Sleeping for %d" % 600
    time.sleep(600)


def main(argv=None):  # pylint: disable=unused-argument

  # if tf.gfile.Exists(FLAGS.eval_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  # tf.gfile.MakeDirs(FLAGS.eval_dir)

  # Read parameters from command line
  opts = read_args(evaluation=True)

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
  model = MainTaggerModel(parameters=parameters, models_path=models_path,
                          overwrite_mappings=opts.overwrite_mappings)
  print "MainTaggerModel location: %s" % model.model_path

  # Data parameters
  lower = parameters['lower']
  zeros = parameters['zeros']
  tag_scheme = parameters['t_s']

  max_sentence_lengths = {}
  max_word_lengths = {}

  # Load sentences
  train_sentences, max_sentence_lengths['train'], max_word_lengths['train'] = \
      loader.load_sentences(opts.train, lower, zeros)
  dev_sentences, max_sentence_lengths['dev'], max_word_lengths['dev'] = loader.load_sentences(
      opts.dev, lower, zeros)
  test_sentences, max_sentence_lengths['test'], max_word_lengths['test'] = loader.load_sentences(
      opts.test, lower, zeros)

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
      sum([x[0] for x in train_stats]), sum([x[0] for x in dev_stats]),
      sum([x[0] for x in test_stats]))

  print "%i / %i / %i longest sentences in train / dev / test." % (
      max([x[0] for x in train_stats]), max([x[0] for x in dev_stats]),
      max([x[0] for x in test_stats]))

  print "%i / %i / %i shortest sentences in train / dev / test." % (
      min([x[0] for x in train_stats]), min([x[0] for x in dev_stats]),
      min([x[0] for x in test_stats]))

  for i, label in [[2, 'char']]:
      print "%i / %i / %i total %s in train / dev / test." % (
          sum([sum(x[i]) for x in train_stats]), sum([sum(x[i]) for x in dev_stats]),
          sum([sum(x[i]) for x in test_stats]),
          label)

      print "%i / %i / %i max. %s lengths in train / dev / test." % (
          max([max(x[i]) for x in train_stats]), max([max(x[i]) for x in dev_stats]),
          max([max(x[i]) for x in test_stats]),
          label)

      print "%i / %i / %i min. %s lengths in train / dev / test." % (
          min([min(x[i]) for x in train_stats]), min([min(x[i]) for x in dev_stats]),
          min([min(x[i]) for x in test_stats]),
          label)

  print "Max. sentence lengths: %s" % max_sentence_lengths
  print "Max. char lengths: %s" % max_word_lengths

  for label, bin_stats, n_unique_words in [['train', train_stats, train_unique_words],
                                           ['dev', dev_stats, dev_unique_words],
                                           ['test', test_stats, test_unique_words]]:
      int32_items = len(train_stats) * (
          max_sentence_lengths[label] * (5 + max_word_lengths[label]) + 1)
      float32_items = n_unique_words * parameters['word_dim']
      total_size = int32_items + float32_items
      logging.info("Input ids size of the %s dataset is %d" % (label, int32_items))
      logging.info("Word embeddings (unique: %d) size of the %s dataset is %d" % (
          n_unique_words, label, float32_items))
      logging.info("Total size of the %s dataset is %d" % (label, total_size))

  batch_size = 5

  # Build the model
  cost, train_step, tag_scores, tag_ids, word_ids, \
  crf_transition_params, sentence_lengths, enqueue_op, placeholders = model.build(
      max_sentence_length_scalar=global_max_sentence_length,
      max_word_length_scalar=global_max_char_length,
      batch_size_scalar=batch_size,
      **parameters)

  evaluate(model,
           dev_buckets, test_buckets,
           opts,
           id_to_tag,
           batch_size,
           placeholders,
           enqueue_op, tag_scores, tag_ids, word_ids, crf_transition_params, sentence_lengths,
           FLAGS.eval_dir,
           tag_scheme)


if __name__ == '__main__':

    tf.app.run()