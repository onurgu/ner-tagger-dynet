{
	total++;
	counts[$2]++;
	if ($2 == n_analyses + 1) {
	  split($3, arr, /+/);
	  split(arr[1], arr2, /'"'"'/);
	  if (length(arr2[2]) == 0) {

		golden_root = arr[1];

		l = 1;
		for (; l < length(golden_root); l++) {

			all_ok = 1;
			for (i=4; i <= NF; i++) {
				split($i, tmp_arr, /+/);
				if (substr(tmp_arr[1], 0, l) != substr(golden_root, 0, l)) {
					all_ok = 0
					break;
				}
			}
			if (all_ok != 1) {
				break;
			}
		}

		if (l-1 == 0) {
			NA_count++;
		} else {
			# print tolower($0);
			# print (l-1);
			output = $1 " " $2;
			for (i=3; i <= NF; i++) {
				output = output " X" substr($i, l);
			}
			print output;
		}
	  };
	  suffix_counts[arr2[2]]++;
	}
} /* END { for (key in counts) { print key, counts[key]; }; for (suffix in suffix_counts) { print suffix, suffix_counts[suffix]; }; print total; } */