#!/usr/bin/env bash

experiment_name=${1:-section1-all-20180311-01}
original_experiment_name=${experiment_name}

ner_tagger_root=/truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/ner-tagger-dynet

#virtualenvwrapper_path=/usr/local/bin/virtualenvwrapper.sh
#virtualenv_name=dynet
virtualenvwrapper_path=/truba/home/ogungor/.local/bin/virtualenvwrapper.sh
virtualenv_name=joint_ner_dynet

#environment_variables_path=environment-variables
environment_variables_path='/truba/sw/centos7.3/comp/intel/PS2017-update1/mkl/bin/mklvars.sh intel64'

datasets_root=/truba/home/ogungor/projects/research/datasets/joint_ner_dynet/

#sacred_args='-m localhost:17017:joint_ner_and_md'
sacred_args='-F /truba/home/ogungor/projects/research/projects/focus/joint_md_and_ner/experiment-logs/'

preamble="cd ${ner_tagger_root} && \
          source ${virtualenvwrapper_path} && \
          workon ${virtualenv_name} && \
          source ${environment_variables_path} && \
          export LD_PRELOAD=/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin/libmkl_avx2.so:/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin/libmkl_def.so:/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin/libmkl_core.so:/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin/libmkl_intel_lp64.so:/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/mkl/lib/intel64_lin/libmkl_intel_thread.so:/truba/sw/centos7.3/comp/intel/PS2017-update1/compilers_and_libraries_2017.1.132/linux/compiler/lib/intel64_lin/libiomp5.so && \
          python control_experiments.py ${sacred_args} with "

dataset_filepaths="datasets_root=${datasets_root} \
					train_filepath=turkish/gungor.ner.train.14.only_consistent \
					dev_filepath=turkish/gungor.ner.dev.14.only_consistent \
					test_filepath=turkish/gungor.ner.test.14.only_consistent "

n_trials=10

dim=${2:-10}
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
		lr_method=sgd-learning_rate_float@0.01 "
		# changed the learning rate to 0.01 from 0.100

		# experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}`
		experiment_name=${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}

		pre_command="echo ${original_experiment_name}-dim-${dim}-morpho_tag_type-${morpho_tag_type}-trial-`printf "%02d" ${trial}` >> ${experiment_name}.log"

		for imode in 0 1 2 ; do
			if [[ $imode == 0 ]]; then
				for amodels in 1 0 ; do
					command=${pre_command}" && "" ${preamble} \
					active_models=${amodels} \
					integration_mode=$imode \
					dynet_gpu=0 \
					embeddings_filepath=\"\" \
					${dataset_filepaths} \
					$small_sizes \
					experiment_name=${experiment_name} ;"
					echo $command;
				done;
				command=${pre_command}" && "" ${preamble} \
				active_models=0 \
				integration_mode=0 \
				use_golden_morpho_analysis_in_word_representation=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				${dataset_filepaths} \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;
			elif [[ $imode == 1 ]]; then
				command=${pre_command}" && "" ${preamble} \
				active_models=2 \
				integration_mode=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				${dataset_filepaths} \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;
			else
				command=${pre_command}" && "" ${preamble} \
				active_models=2 \
				integration_mode=2 \
				multilayer=1 \
				shortcut_connections=1 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				${dataset_filepaths} \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;

				command=${pre_command}" && "" ${preamble} \
				active_models=2 \
				integration_mode=2 \
				dynet_gpu=0 \
				embeddings_filepath=\"\" \
				${dataset_filepaths} \
				$small_sizes \
				experiment_name=${experiment_name} ;"
				echo $command;

			fi ;
		done

	done
done