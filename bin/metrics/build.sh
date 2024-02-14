#!/bin/bash
# =============================================================================
# Shared objects
#
# Objective: Build compiled shared objects (*.so files) from *pyx files.
#
# Version: 0.1.0
#
# Author: Diptesh
#
# Date: Apr 17, 2020
#
# =============================================================================

python setup.py build_ext --inplace

pat="\.\/[^\/]+\.so"

for i in $(find -name "*.so")
do
    if [[ $i =~ $pat ]]
    then
        file_new=$(sed -E 's/(\.\/)([a-z0-9]+)(\..+\.so)/\2.so/' <<< $i)
        file_old=$(sed -E 's/(\.\/)(.+)/\2/' <<< $i)
    	mv $file_old $file_new
	fi
done
