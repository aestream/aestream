#!/usr/bin/env bash

# Thanks to https://stackoverflow.com/a/17841619
function join_by {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

AESTREAM_PATHS=$(find /opt -name 'aestream' -type d | grep lib)
INSTALL_PATHS=`join_by ':' $AESTREAM_PATHS`
echo "Setting LD_LIBRARY_PATH to ${INSTALL_PATHS}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${INSTALL_PATHS}"
