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
		word_lstm_dim=$dim \
		lr_method=sgd-learning_rate_float@0.100 "

		# experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}`
		experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}

		pre_command="echo ${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}` >> ${experiment_name}.log"

		for imode in 0 1 2 ; do
			if [[ $imode == 0 ]]; then
				for amodels in 1 0 ; do
					command=${pre_command}" && ""cd ${ner_tagger_root} && \
					source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
					source environment-variables && python control_experiments.py -m joint_ner_and_md with \
					active_models=${amodels} \
					integration_mode=$imode \
					dynet_gpu=0 \
					embeddings_filepath=\"\" \
					train_filepath=turkish/gungor.ner.train.14.only_consistent \
					dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
					test_filepath=turkish/gungor.ner.test.14.only_consistent \
					$small_sizes \
					experiment_name=${experiment_name} ;"
					echo $command;
				done;
				command=${pre_command}" && ""cd ${ner_tagger_root} && \
				source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
				source environment-variables && python control_experiments.py -m joint_ner_and_md with \
				active_models=0 \
				integration_mode=0 \
				use_golden_morpho_analysis_in_word_representation=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				train_filepath=turkish/gungor.ner.train.14.only_consistent \
				dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
				test_filepath=turkish/gungor.ner.test.14.only_consistent \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;
			elif [[ $imode == 1 ]]; then
				command=${pre_command}" && ""cd ${ner_tagger_root} && \
				source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
				source environment-variables && python control_experiments.py -m joint_ner_and_md with \
				active_models=2 \
				integration_mode=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				train_filepath=turkish/gungor.ner.train.14.only_consistent \
				dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
				test_filepath=turkish/gungor.ner.test.14.only_consistent \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;
			else
				command=${pre_command}" && ""cd ${ner_tagger_root} && \
				source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
				source environment-variables && python control_experiments.py -m joint_ner_and_md with \
				active_models=2 \
				integration_mode=2 \
				multilayer=1 \
				shortcut_connections=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				train_filepath=turkish/gungor.ner.train.14.only_consistent \
				dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
				test_filepath=turkish/gungor.ner.test.14.only_consistent \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;

				command=${pre_command}" && ""cd ${ner_tagger_root} && \
				source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
				source environment-variables && python control_experiments.py -m joint_ner_and_md with \
				active_models=2 \
				integration_mode=2 \
				dynet_gpu=0 \
				embeddings_filepath="" \
				train_filepath=turkish/gungor.ner.train.14.only_consistent \
				dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
				test_filepath=turkish/gungor.ner.test.14.only_consistent \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;

			fi ;
		done

	done
done