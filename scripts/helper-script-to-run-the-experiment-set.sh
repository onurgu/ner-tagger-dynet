#!/usr/bin/env bash

experiment_name=${1:-section1-all-20171114-08}

ner_tagger_root=/home/onur/projects/research/focus/ner-tagger-dynet-multilayer

for imode in 0 1 2 ; do
	if [[ $imode == 0 ]]; then
		for amodels in 1 0 ; do
			cd ${ner_tagger_root} && \
			source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
			source environment-variables && python control_experiments.py -m joint_ner_and_md with \
			active_models=${amodels} \
			integration_mode=$imode \
			dynet_gpu=0 \
			embeddings_filepath="" \
			train_filepath=turkish/gungor.ner.train.14.only_consistent \
			dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
			test_filepath=turkish/gungor.ner.test.14.only_consistent \
			experiment_name=${experiment_name} ;
		done;
		cd ${ner_tagger_root} && \
		source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
		source environment-variables && python control_experiments.py -m joint_ner_and_md with \
		active_models=0 \
		integration_mode=0 \
		use_golden_morpho_analysis_in_word_representation=1 \
		dynet_gpu=0 \
		embeddings_filepath="" \
		train_filepath=turkish/gungor.ner.train.14.only_consistent \
		dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
		test_filepath=turkish/gungor.ner.test.14.only_consistent \
		experiment_name=${experiment_name} ;
	elif [[ $imode == 1 ]]; then
		cd ${ner_tagger_root} && \
		source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
		source environment-variables && python control_experiments.py -m joint_ner_and_md with \
		active_models=2 \
		integration_mode=1 \
		dynet_gpu=0 \
		embeddings_filepath="" \
		train_filepath=turkish/gungor.ner.train.14.only_consistent \
		dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
		test_filepath=turkish/gungor.ner.test.14.only_consistent \
		experiment_name=${experiment_name} ;
	else
		cd ${ner_tagger_root} && \
		source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
		source environment-variables && python control_experiments.py -m joint_ner_and_md with \
		active_models=2 \
		integration_mode=2 \
		multilayer=1 \
		shortcut_connections=1 \
		dynet_gpu=0 \
		embeddings_filepath="" \
		train_filepath=turkish/gungor.ner.train.14.only_consistent \
		dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
		test_filepath=turkish/gungor.ner.test.14.only_consistent \
		experiment_name=${experiment_name} ;

		cd ${ner_tagger_root} && \
		source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
		source environment-variables && python control_experiments.py -m joint_ner_and_md with \
		active_models=2 \
		integration_mode=2 \
		dynet_gpu=0 \
		embeddings_filepath="" \
		train_filepath=turkish/gungor.ner.train.14.only_consistent \
		dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
		test_filepath=turkish/gungor.ner.test.14.only_consistent \
		experiment_name=${experiment_name} ;

	fi ;
done