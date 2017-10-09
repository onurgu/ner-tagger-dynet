from sacred import Experiment

from sacred.observers import MongoObserver

import subprocess
import sys
import re

ex = Experiment('my_experiment')

@ex.config
def my_config():
    skip_testing = 0
    reload = 0
    max_epochs = 50

    dynet_gpu = 0

    host="localhost"
    experiment_name = "default_experiment_name"

    datasets_root = "/home/onur/projects/research/turkish-ner/datasets"

    learning_rate = 0.01

    crf = 1
    lr_method = "sgd-learning_rate_float@%lf" % learning_rate
    dropout = 0.5
    char_dim = 128
    char_lstm_dim = 128

    morpho_tag_dim = 128
    morpho_tag_lstm_dim = 128
    morpho_tag_type = "wo_root"

    morpho_tag_column_index = 1

    integration_mode = 0

    word_dim = 128
    word_lstm_dim = 128
    cap_dim = 0

    # char_dim = 200
    # char_lstm_dim = 200
    #
    # morpho_tag_dim = 100
    # morpho_tag_lstm_dim = 200
    # morpho_tag_type = "wo_root"
    #
    # morpho_tag_column_index = 1
    #
    # integration_mode = 0
    #
    # word_dim = 300
    # word_lstm_dim = 200
    # cap_dim = 100

    train_filepath = "turkish/gungor.ner.train.only_consistent"
    dev_filepath = "turkish/gungor.ner.dev.only_consistent"
    test_filepath = "turkish/gungor.ner.test.only_consistent"

    yuret_train_filepath = "turkish/train.merge.utf8.gungor_format"
    yuret_test_filepath = "turkish/test.merge.utf8.gungor_format"

    embeddings_filepath = "turkish/we-300.txt"


@ex.main
def my_main():

    run_a_single_configuration_without_fabric()

from utils import read_args, form_parameters_dict, get_name, get_model_subpath


@ex.capture
def run_a_single_configuration_without_fabric(
                                              datasets_root,
                                              crf,
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
                                              cap_dim, skip_testing, max_epochs,
                                              train_filepath,
                                              dev_filepath,
                                              test_filepath,
                                              yuret_train_filepath,
                                              yuret_test_filepath,
                                              embeddings_filepath,
                                              integration_mode,
                                              reload,
                                              dynet_gpu,
                                              _run):

    """
    python train.py --pre_emb ../../data/we-300.txt --train dataset/gungor.ner.train.only_consistent --dev dataset/gungor.ner.dev.only_consistent --test dataset/gungor.ner.test.only_consistent --word_di
m 300  --word_lstm_dim 200 --word_bidirect 1 --cap_dim 100 --crf 1 --lr_method=sgd-learning_rate_float@0.05 --maximum-epochs 50 --char_dim 200 --char_lstm_dim 200 --char_bid
irect 1 --overwrite-mappings 1 --batch-size 1 --morpho_tag_dim 100 --integration_mode 2
    """

    execution_part = "python train.py --overwrite-mappings 1 "
    if dynet_gpu == 1:
        execution_part += "--dynet-gpu 1 "

    if word_dim == 0:
        embeddings_part = ""
    else:
        if embeddings_filepath:
            embeddings_part = "--pre_emb %s/%s " % (datasets_root, embeddings_filepath)
        else:
            embeddings_part = ""

    print (train_filepath, dev_filepath, test_filepath, skip_testing, max_epochs)

    always_constant_part = "-T %s/%s " \
          "-d %s/%s " \
          "-t %s/%s " \
          "--yuret_train %s/%s " \
          "--yuret_test %s/%s " \
          "%s" \
          "--skip-testing %d " \
          "--tag_scheme iobes " \
          "--maximum-epochs %d " % (datasets_root, train_filepath,
                                    datasets_root, dev_filepath,
                                    datasets_root, test_filepath,
                                    datasets_root, yuret_train_filepath,
                                    datasets_root, yuret_test_filepath,
                                    embeddings_part, skip_testing, max_epochs)

    commandline_args = always_constant_part + \
              "--crf %d " \
              "--lr_method %s " \
              "--dropout %1.1lf " \
              "--char_dim %d " \
              "--char_lstm_dim %d " \
              "--morpho_tag_dim %d " \
              "--morpho-tag-column-index %d " \
              "--word_dim %d " \
              "--word_lstm_dim %d "\
              "--cap_dim %d "\
              "--integration_mode %d " \
              "--reload %d" % (crf,
                               lr_method,
                               dropout,
                               char_dim,
                               char_lstm_dim,
                               morpho_tag_dim,
                               morpho_tag_column_index,
                               word_dim,
                               word_lstm_dim,
                               cap_dim,
                               integration_mode,
                               reload)

    # tagger_root = "/media/storage/genie/turkish-ner/code/tagger"

    print _run
    print _run.info

    print subprocess.check_output(["id"])
    print subprocess.check_output(["pwd"])

    opts = read_args(args_as_a_list=commandline_args.split(" "))
    print opts
    parameters = form_parameters_dict(opts)
    print parameters
    # model_path = get_name(parameters)
    model_path = get_model_subpath(parameters)
    print model_path

    task_names = ["NER", "MORPH", "YURET"]

    for task_name in task_names:
        _run.info["%s_dev_f_score" % task_name] = dict()
        _run.info["%s_test_f_score" % task_name] = dict()

    _run.info['starting'] = 1

    dummy_prefix = ""

    print dummy_prefix + execution_part + commandline_args
    process = subprocess.Popen((dummy_prefix + execution_part + commandline_args).split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def record_metric(epoch, label, value):
        if str(epoch) in _run.info[label]:
            _run.info[label][str(epoch)].append(value)
        else:
            _run.info[label][str(epoch)] = list()
            _run.info[label][str(epoch)].append(value)

    def capture_information(line):

        # 1
        """
        NER Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf 
        """
        for task_name in task_names:
            m = re.match("^%s Epoch: (\d+) .* best_dev, best_test: (.+) (.+)$" % task_name, line)
            if m:
                epoch = int(m.group(1))
                best_dev = float(m.group(2))
                best_test = float(m.group(3))

                record_metric(epoch, "%s_dev_f_score" % task_name, best_dev)
                record_metric(epoch, "%s_test_f_score" % task_name, best_test)

    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        capture_information(line)
        sys.stdout.flush()

    return model_path

if __name__ == '__main__':
    ex.run_commandline()