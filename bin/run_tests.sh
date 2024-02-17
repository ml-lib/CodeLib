#!/bin/bash
# =============================================================================
# Python unit tests and code ratings
#
# Objective: Test all python test modules and rate all python scripts
#
# Version: 0.2.0
#
# Author: Diptesh
#
# Date: Jun 24, 2020
#
# =============================================================================

# Set test directory
path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

test_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1\/tests/' <<< $path)
proj_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1/' <<< $path)

if [[ -z $1 ]]
then
	module="-a"
else
	module=$1
fi

printf "=%.0s" {1..70}

# Run unit tests
if [[ $module == "-a" || $module == "-u" ]]
then
	printf "\nRunning unit & integration tests ...\n\n"
	coverage run -m unittest discover -v -s $test_dir -p "test_*.py"	
  printf "=%.0s" {1..70}
	printf "\n"
	printf "\nComputing test coverage ...\n\n"
	coverage report -m --omit="*/tests/test_*,*/opt/spark-*" 2>&1 | tee "$proj_dir/logs/cov.out"
	COV_SCORE=`grep "TOTAL" $proj_dir/logs/cov.out | tail -1 | awk '{ printf("%d", $4) }'`
	COV_COLOR="red"
	if [[ $COV_SCORE == "100" ]]
	then
		COV_COLOR="dagreen"
	fi
	sed -i "3s/.*/\[\!\[Coverage score\]\(\https\:\/\/img\.shields\.io\/badge\/coverage\-$COV_SCORE\%25\-$COV_COLOR.svg\)\]\(\.\/logs\/cov\.out\)/" "$proj_dir/README.md"
	printf "=%.0s" {1..70}
fi

# Rate coding styles for all python scripts
if [[ $module == "-a" || $module == "-r" ]]
then
    printf "\nRating code style ...\n\n"
    score=0
    cnt=0
		rm $proj_dir/logs/pylint/*.out
    for i in $(find "$proj_dir" -name "*.py")
    do
        file=${i#$(dirname "$(dirname "$i")")/}
        printf "%-67s %s" "$file"
        file_dir=$(sed -E 's/(.+\/)(.+\.py)/\1/' <<< $i)
        cd "$file_dir"
        pylint "$i" > "$proj_dir/logs/pylint/pylint.out"
        PYLINT_SCORE=`grep "Your code has been rated" $proj_dir/logs/pylint/pylint.out | cut -d" " -f7 | cut -d"." -f1`
        file_name=$(sed -E 's/(\/)/-/' <<< $file)
        file_name=$(sed -E 's/(\.)/-/' <<< $file_name)
        cp "$proj_dir/logs/pylint/pylint.out" "$proj_dir/logs/pylint/$file_name.out"
        score=$((score + PYLINT_SCORE))
        cnt=$((cnt + 1))
        printf "$PYLINT_SCORE\n"
        cd "$proj_dir"
    done
    tot_score=$(echo "scale=1; $score/$cnt" | bc)
    printf "\nTotal score: $tot_score\n"
    # Add pylint badge to README.md
    sed -i "2s/.*/\[\!\[pylint Score\]\(https\:\/\/mperlet\.github\.io\/pybadge\/badges\/$tot_score.svg\)\]\(\.\/logs\/pylint\/\)/" "$proj_dir/README.md"
    printf "=%.0s" {1..70}
    printf "\n"
fi

pipreqs --force --use-local $proj_dir &> $proj_dir/logs/pip.out

exit 0
