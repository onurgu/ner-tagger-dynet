#!/usr/bin/env bash

EXP_NAME=${1:-default_exp_name}
GPU=${2:-0}

echo 'cd /home/onur/projects/research/focus/ner-tagger-tensorflow && source /usr/local/bin/virtualenvwrapper.sh && workon dynet && source environment-variables && python control_experiments.py -m joint_ner_and_md with integration_mode=0 dynet_gpu='$GPU' embeddings_filepath="" word_lstm_dim=256 experiment_name='$EXP_NAME
echo 'cd /home/onur/projects/research/focus/ner-tagger-tensorflow && source /usr/local/bin/virtualenvwrapper.sh && workon dynet && source environment-variables && python control_experiments.py -m joint_ner_and_md with integration_mode=1 dynet_gpu='$GPU' embeddings_filepath="" word_lstm_dim=256 experiment_name='$EXP_NAME
echo 'cd /home/onur/projects/research/focus/ner-tagger-tensorflow && source /usr/local/bin/virtualenvwrapper.sh && workon dynet && source environment-variables && python control_experiments.py -m joint_ner_and_md with integration_mode=2 dynet_gpu='$GPU' embeddings_filepath="" word_lstm_dim=256 experiment_name='$EXP_NAME