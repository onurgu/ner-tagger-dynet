#!/usr/bin/env bash

export experiment_name=section1-all-20171114-08

for imode in 0 1 2 ; do
	if [[ $imode == 0 ]]; then
		for amodels in 1 0 ; do
			cd /home/onur/projects/research/focus/ner-tagger-tensorflow && \
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
	else
		cd /home/onur/projects/research/focus/ner-tagger-tensorflow && \
		source /usr/local/bin/virtualenvwrapper.sh && workon dynet && \
		source environment-variables && python control_experiments.py -m joint_ner_and_md with \
		active_models=2 \
		integration_mode=$imode \
		dynet_gpu=0 \
		embeddings_filepath="" \
		train_filepath=turkish/gungor.ner.train.14.only_consistent \
		dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
		test_filepath=turkish/gungor.ner.test.14.only_consistent \
		experiment_name=${experiment_name} ;
	fi ;
done