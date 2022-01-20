#!/bin/bash
# =============================================================================
# Installation file
#
# Version: 0.2.0
#
# Author: Diptesh
#
# Date: May 03, 2020
#
# =============================================================================

# =============================================================================
# DO NOT CHANGE ANYTHING FROM HERE
# =============================================================================

error=0
__version__="0.4.0"

# =============================================================================
# User defined functions
# =============================================================================

mod()
{
  exec=$1
  file=$2
  state="[ OK ]"
  if ! $exec $file; then
    state="[fail]"
    error=$((error + 1))
  fi
  printf "%-72s %s\n" "$2" "$state"
}

# =============================================================================
# Main
# =============================================================================

path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\
/$(basename "${BASH_SOURCE[0]}")"

proj_dir=$(sed -E 's/(.+\/)(.+)/\1/' <<< $path)

printf "Installing v$__version__ ...\n\n"

for i in $(find "$proj_dir" -maxdepth 20 -name "*.sh")
do
  file_name=${i#$proj_dir}
  mod "chmod +x" "$file_name"
  if [[ "$file_name" == "programs.sh" ]]; then
    bash bin/programs.sh
  fi
done

state="[Done]"
if [[ $error -gt 0 ]]; then
  state="[fail]"
fi

printf "%-72s %s\n" "Installation" "$state"

exit $error
