#!/usr/bin/env bash

declare -A n_consistent_lines
declare -A n_consistent_sentences

for label in train dev test ; do
	cp dataset/gungor.ner.${label} dataset/gungor.ner.${label}.01

	echo starting sentences $label $(cat dataset/gungor.ner.${label}.01 | awk '/^$/ { count += 1 } END { print count }')

	cat dataset/gungor.ner.${label}.01 | bash ./scripts/strip-sentences-with-inconsistent-morph-analysis.sh > dataset/gungor.ner.${label}.01.only_consistent ;
	n_consistent_lines[$label]=$(wc -l dataset/gungor.ner.${label}.01.only_consistent | cut -d" " -f1)
	n_consistent_sentences[$label]=$(cat dataset/gungor.ner.${label}.01.only_consistent | awk '/^$/ { count += 1 } END { print count }')
	# printf "%s %02d %6d %6d\n" $label 0 ${n_consistent_lines[$label]} ${n_consistent_sentences[$label]}
done

printf "level | n_lines_changed n_consistent_lines n_consistent_sentences | ...\n"
printf "%02d | train %6d %6d %6d | dev %6d %6d %6d | test %6d %6d %6d\n" -1 \
		-1 ${n_consistent_lines['train']} ${n_consistent_sentences['train']} \
		-1 ${n_consistent_lines['dev']} ${n_consistent_sentences['dev']} \
		-1 ${n_consistent_lines['test']} ${n_consistent_sentences['test']}

#echo 'continue...'
#read

for n_analyzes in `seq 1 13`; do

	declare -A n_lines_changed
	declare -A n_consistent_lines
	declare -A n_consistent_sentences

	if [[ -f Xoutput-n_analyses-`printf "%02d" ${n_analyzes}`.txt.rules ]]; then
		cat Xoutput-n_analyses-`printf "%02d" ${n_analyzes}`.txt.rules | \
		  awk '{ output = ""; for (i=2; i < NF; i++) { if (i == 2) { replacement_pattern= "\\(.\\+\\)"; } else { replacement_pattern = "\\3"; }; gsub(/^X/, replacement_pattern, $i); if (i == 2) { output = "s/^\\(.\\+\\) \\(" $i "\\)\\("; } else { output = output " " $i; } }; gsub(/^X/, "\\3", $NF); output = output " .\\+\\)$/\\1 " $NF "\\4/g"; print output; }' > Xoutput-n_analyses-`printf "%02d" ${n_analyzes}`.txt.rules.sed ;

		for label in train dev test ; do
#			 echo $label
			if [[ ${n_analyzes} == 1 ]]; then
				cat dataset/gungor.ner.${label}.`printf "%02d" ${n_analyzes}` | awk 'NF == 4 { print $1, $3, $3, $4 } NF != 4 { print }' > dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`
				n_lines_changed[$label]=-1
			else
				sed -f Xoutput-n_analyses-`printf "%02d" ${n_analyzes}`.txt.rules.sed dataset/gungor.ner.${label}.`printf "%02d" ${n_analyzes}` > dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`
				n_lines_changed[$label]=$(($(diff dataset/gungor.ner.${label}.`printf "%02d" ${n_analyzes}` dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))` | wc -l)/4))
				# echo ${n_lines_changed[$label]}
			fi

			cat dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))` | bash ./scripts/strip-sentences-with-inconsistent-morph-analysis.sh > dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent ;
			n_consistent_lines[$label]=$(wc -l dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent | cut -d" " -f1)
			n_consistent_sentences[$label]=$(cat dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent | awk '/^$/ { count += 1 } END { print count }')
			# echo ${n_consistent_lines[$label]}
		done

		if [[ ${n_analyzes} == 1 ]]; then
			printf "ran awk rule on 01 level to get the %02d level data files\n" $((n_analyzes)) $((n_analyzes+1))
		else
			printf "ran sed rules on %02d level to get the %02d level data files\n" $((n_analyzes)) $((n_analyzes+1))
		fi

		printf "level | n_lines_changed n_consistent_lines n_consistent_sentences | ...\n"
		printf "%02d | train %6d %6d %6d | dev %6d %6d %6d | test %6d %6d %6d\n" ${n_analyzes} \
		${n_lines_changed['train']} ${n_consistent_lines['train']} ${n_consistent_sentences['train']} \
		${n_lines_changed['dev']} ${n_consistent_lines['dev']} ${n_consistent_sentences['dev']} \
		${n_lines_changed['test']} ${n_consistent_lines['test']} ${n_consistent_sentences['test']}
	else
		for label in train dev test ; do
			cp dataset/gungor.ner.${label}.`printf "%02d" ${n_analyzes}` dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`

			cat dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))` | bash ./scripts/strip-sentences-with-inconsistent-morph-analysis.sh > dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent ;
			n_consistent_lines[$label]=$(wc -l dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent | cut -d" " -f1)
			n_consistent_sentences[$label]=$(cat dataset/gungor.ner.${label}.`printf "%02d" $((n_analyzes+1))`.only_consistent | awk '/^$/ { count += 1 } END { print count }')
			# echo ${n_consistent_lines[$label]}
		done
#		printf "%02d | %6d %6d %6d | %6d %6d %6d\n" ${n_analyzes} 0 0 0 0 0 0
		printf "NO SED RULES FOUND. only copied %02d level data files to get the %02d level data files\n" $((n_analyzes)) $((n_analyzes+1))
		printf "level | n_lines_changed n_consistent_lines n_consistent_sentences | ...\n"
		printf "%02d | train %6d %6d %6d | dev %6d %6d %6d | test %6d %6d %6d\n" ${n_analyzes} \
			0 ${n_consistent_lines['train']} ${n_consistent_sentences['train']} \
			0 ${n_consistent_lines['dev']} ${n_consistent_sentences['dev']} \
			0 ${n_consistent_lines['test']} ${n_consistent_sentences['test']}
	fi

#	echo 'continue...'
#	read

done

