#!/bin/bash
# =============================================================================
# Python dependencies
#
# Objective: Install python dependencies from requirements.txt
#
# Version: 0.1.0
#
# Author: Diptesh
#
# Date: Mar 03, 2020
#
# =============================================================================

# Set test directory
path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

proj_dir=$(sed -E 's/(.+)(\/bin\/.+)/\1/' <<< $path)

if ! [[ $(hostname) =~ .+target\.com ]]; then
    pip install -r $proj_dir/requirements.txt
fi
