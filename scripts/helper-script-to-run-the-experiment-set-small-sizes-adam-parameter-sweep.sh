#!/usr/bin/env bash

experiment_name=${1:-section1-all-20171114-08}
original_experiment_name=${experiment_name}

ner_tagger_root=/home/onur/projects/research/focus/ner-tagger-dynet-multilayer

n_trials=10

dim=10
morpho_tag_type=char

for trial in `seq 1 ${n_trials}`; do

	for morpho_tag_type in char ; do

		small_sizes="char_dim=$dim \
		char_lstm_dim=$dim \
		morpho_tag_dim=$dim \
		morpho_tag_lstm_dim=$dim \
		morpho_tag_type=${morpho_tag_type} \
		word_dim=$dim \
		word_lstm_dim=$dim "

		# experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}`
		experiment_name=${original_experiment_name}-adam-sweep-sparse_updates_enabled

		for learning_rate in 0.1 0.05 0.01 0.005 0.001 ; do
			for sparse_updates_enabled in 0 ; do
				pre_command="echo ${original_experiment_name}-learning_rate-${learning_rate}-trial-`printf "%02d" ${trial}` >> ${experiment_name}.log"
				command=${pre_command}" && ""cd ${ner_tagger_root} && \
				source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
				source environment-variables && python control_experiments.py -m joint_ner_and_md with \
				active_models=0 \
				integration_mode=0 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				train_filepath=turkish/gungor.ner.train.14.only_consistent \
				dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
				test_filepath=turkish/gungor.ner.test.14.only_consistent \
				$small_sizes \
				lr_method=`printf "adam-alpha_float@%.03lf" ${learning_rate}` \
				sparse_updates_enabled=${sparse_updates_enabled}
				max_epochs=10 \
				experiment_name=${experiment_name} ;"
				echo $command;
			done;
		done
	done
done