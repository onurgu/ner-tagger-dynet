from sacred import Experiment

# from fabric.api import *
# import fabric.tasks

# env.use_ssh_config = True
# env.parallel = True

hosts = ['genie.01', 'genie.02']

# @task
# def get_id():
#     run("id")

import subprocess
import sys
import re

ex = Experiment('my_experiment')
import os

@ex.config
def my_config():
    skip_testing = 0
    reload = 0
    max_epochs = 100

    host="localhost"
    experiment_name = "default_experiment_name"

    learning_rate = 0.01

    crf = 1
    lr_method = "sgd-lr_%lf" % learning_rate
    dropout = 0.5
    char_dim = 200
    char_lstm_dim = 200
    morpho_tag_dim = 200
    morpho_tag_lstm_dim = 200
    morpho_tag_type = "wo_root"
    word_dim = 100
    word_lstm_dim = 300
    cap_dim = 100
    separate_bilstms = 0

    morpho_tag_column_index = 1

    train_filepath = "GokhanTurVeri/train-iob"
    dev_filepath = "GokhanTurVeri/test-iob"
    test_filepath = "GokhanTurVeri/test-iob"

    embeddings_filepath = "word2vec/huawei-corpus/we-100.txt"

def enumerate_every_possible_configuration_update():
    values_to_try = dict()
    # values_to_try['learning_rate'] = [0.01, 0.05]
    values_to_try['lr_method'] = ["sgd-lr_0.01", "sgd-lr_0.05"]
    values_to_try['dropout'] = [0.25, 0.5, 0.75]
    values_to_try['char_dim'] = [0, 50, 100, 200, 400]
    values_to_try['char_lstm_dim'] = [25, 50, 100, 200, 400]
    values_to_try['morpho_tag_dim'] = [0, 50, 100, 200, 400]
    values_to_try['morpho_tag_lstm_dim'] = [25, 50, 100, 200, 400]
    values_to_try['morpho_tag_type'] = ['wo_root', 'wo_root_after_DB',
                                       'with_root', 'with_root_after_DB',
                                        'char']

    # values_to_try['word_dim'] = [0, 50, 100, 200, 400]
    values_to_try['word_dim'] = [100]
    values_to_try['word_lstm_dim'] = [25, 50, 100, 200, 400]
    values_to_try['cap_dim'] = [0, 50, 100, 200, 400]

    for parameter_name in values_to_try.keys():
        if len(values_to_try[parameter_name]) == 1:
            continue
        for parameter_value in values_to_try[parameter_name]:
            yield {parameter_name: parameter_value}

@ex.main
def my_main():

    run_a_single_configuration_without_fabric()

from utils import read_args, form_parameters_dict, get_name, get_model_subpath


@ex.capture
def run_a_single_configuration_without_fabric(crf,
                                              lr_method,
                                              dropout,
                                              char_dim,
                                              char_lstm_dim,
                                              morpho_tag_dim,
                                              morpho_tag_lstm_dim,
                                              morpho_tag_type,
                                              morpho_tag_column_index,
                                              word_dim,
                                              word_lstm_dim,
                                              cap_dim, separate_bilstms, skip_testing, max_epochs,
                                              train_filepath,
                                              dev_filepath,
                                              test_filepath,
                                              embeddings_filepath,
                                              reload,
                                              _run):

    from sacred.observers import MongoObserver

    """
    python train.py --pre_emb ../../data/we-300.txt --train dataset/tr.train --dev dataset/tr.test --test dataset/tr.test --word_dim 300  --word_lstm_dim 200 --word_bidirect 1 --cap_dim 100 --crf 1 --lr_method=sgd-lr_0.01 --maximum-epochs 100 --char_dim 200 --char_lstm_dim 200 --char_bidirect 1 --morpho_tag_dim 100 --morpho_tag_lstm_dim 100 --morpho_tag_type char --overwrite-mappings 1 --batch-size 5
    """

    execution_part = "python train.py "

    if word_dim == 0:
        embeddings_part = ""
    else:
        embeddings_part = "--pre_emb ../../datasets/%s " % embeddings_filepath

    print (train_filepath, dev_filepath, test_filepath, skip_testing, max_epochs)

    always_constant_part = "-T ../../datasets/%s " \
          "-d ../../datasets/%s " \
          "-t ../../datasets/%s " \
          "%s" \
          "--skip-testing %d " \
          "--tag_scheme iobes " \
          "--maximum-epochs %d " % (train_filepath, dev_filepath, test_filepath, embeddings_part, skip_testing, max_epochs)

    commandline_args = always_constant_part + \
              "--crf %d " \
              "--lr_method %s " \
              "--dropout %1.1lf " \
              "--char_dim %d " \
              "--char_lstm_dim %d " \
              "--morpho_tag_dim %d " \
              "--morpho_tag_lstm_dim %d " \
              "--morpho_tag_type %s " \
              "--morpho-tag-column-index %d " \
              "--word_dim %d " \
              "--word_lstm_dim %d "\
              "--cap_dim %d "\
              "--separate-bilstms %d "\
              "--reload %d" % (crf,
                               lr_method,
                               dropout,
                               char_dim,
                               char_lstm_dim,
                               morpho_tag_dim,
                               morpho_tag_lstm_dim,
                               morpho_tag_type,
                               morpho_tag_column_index,
                               word_dim,
                               word_lstm_dim,
                               cap_dim,
                               separate_bilstms,
                               reload)

    tagger_root = "/media/storage/genie/turkish-ner/code/tagger"

    print _run
    print _run.info

    print subprocess.check_output(["id"])
    print subprocess.check_output(["pwd"])

    opts = read_args(commandline_args.split(" "))
    # print opts
    parameters = form_parameters_dict(opts)
    # print parameters
    # model_path = get_name(parameters)
    model_path = get_model_subpath(parameters)
    print model_path


    _run.info['costs'] = dict()
    _run.info['best_performances'] = dict()

    _run.info['starting'] = 1

    dummy_prefix = ""

    print dummy_prefix + execution_part + commandline_args
    process = subprocess.Popen((dummy_prefix + execution_part + commandline_args).split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def record_metric(_run, epoch, samples, label, value):
        if str(epoch) in _run.info[label]:
            _run.info[label][str(epoch)].append(value)
        else:
            _run.info[label][str(epoch)] = list()
            _run.info[label][str(epoch)].append(value)

    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        m = re.match("^Epoch (\d+): (\d+) Samples read. Avg. cost: ([^,]+), Scores on dev: ([^,]+), (.+)$", line)
        if m:
            epoch = int(m.group(1))
            samples = int(m.group(2))
            epoch_avg_cost = float(m.group(3))
            if skip_testing == 1 or dev_filepath == test_filepath:
                epoch_performance = float(m.group(4))
            else:
                epoch_performance = float(m.group(5))
            record_metric(_run, epoch, samples, "costs", epoch_avg_cost)
            record_metric(_run, epoch, samples, "best_performances", epoch_performance)
        sys.stdout.flush()

    # for epoch in range(max_epochs):
    #     epoch_cost = subprocess.check_output(("tail -1 %s" % os.path.join("models", model_path, "epoch-%08d" % epoch, "epoch_cost.txt")).split(" "))
    #     best_performances = subprocess.check_output(("cat %s" % os.path.join("models", model_path, "epoch-%08d" % epoch, "best_performances.txt")).split(" "))
    #     print "EPOCHCOST: " + epoch_cost
    #     _run.info['costs'][str(epoch)] = float(epoch_cost.strip())
    #     print "BESTPERF: " + best_performances
    #     if skip_testing == 1 or dev_filepath == test_filepath:
    #         _run.info['best_performances'][str(epoch)] = float(best_performances.split(" ")[0])
    #     else:
    #         _run.info['best_performances'][str(epoch)] = float(best_performances.split(" ")[1])

    return model_path

if __name__ == '__main__':
    ex.run_commandline()