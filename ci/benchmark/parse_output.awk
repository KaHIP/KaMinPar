BEGIN {
	split("", data)
}

match($0, /RESULT cut=([0-9]+) imbalance=([0-9\.\-e]+) feasible=(0|1)/, m) {
	data["Cut"] = m[1]
	data["Imbalance"] = m[2]
	data["Feasible"] = m[3]
}

match($0, /G\.partitioning=([0-9\.\-e]+)/, m) {
	data["Time"] = m[1]
}

END {
	OK = (length(data) == 4)
	printf "%d,%f,%d,%f,%d\n", 
	       data["Cut"],
	       data["Imbalance"],
	       data["Feasible"],
	       data["Time"],
	       OK
}
