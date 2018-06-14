#!/usr/bin/env bash

echo $0
rundir_path=`dirname $0`
experiment_name=${1:-TRUBA-all-experiments-20180311-01}
dim=${2:-10}

partition_name=${3:-short}
core_per_job=${4:-4}
max_time=${5:-4-00:00:00}

sub_job_id=0
max_jobs_to_submit=100

# jobs_line_by_line=`${rundir_path}/helper-script-to-run-the-experiment-set-TRUBA.sh ${experiment_name} ${dim}`

#echo $jobs_line_by_line | while read line; do

${rundir_path}/helper-script-to-run-the-experiment-set-TRUBA.sh ${experiment_name} ${dim} | while read line; do

	sub_job_id=$((sub_job_id + 1))
	echo $sub_job_id
	echo $max_jobs_to_submit
	echo $line

	# experiment_name=XXX-dim-10-morpho_tag_type-char
	job_id=`echo ${line} | awk '{ match($0, /.* experiment_name=([^ ]+) /, arr); printf "%s", arr[1]; }'`

	echo '#!/bin/bash' > ${rundir_path}/batch-script-${job_id}.sh
	echo $line >> ${rundir_path}/batch-script-${job_id}.sh

	sbatch -A ogungor -J ${job_id} -p ${partition_name} -c ${core_per_job} --time=${max_time} --mail-type=END --mail-user=onurgu@boun.edu.tr ${rundir_path}/batch-script-${job_id}.sh

	echo sleeping for 120 seconds to allow time to FileStorageObserver
	sleep 120

	if [[ sub_job_id -eq max_jobs_to_submit ]]; then
		# echo exit
		exit
	fi
done