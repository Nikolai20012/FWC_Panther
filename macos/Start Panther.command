#!/usr/bin/env bash
# Double-click this in Finder to set up (first time) and start the Panther
# Detector. It just hands off to the shell script next to it.
#
# If double-clicking does nothing, the executable bit was lost (zip does that).
# Open Terminal and run:   chmod +x "macos/Start Panther.command" macos/start-panther.sh

DIR=$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "$DIR/start-panther.sh" "$@"
status=$?

if [ $status -ne 0 ]; then
  echo
  echo 'Setup or startup failed. Scroll up for the first error.'
  read -r -p 'Press Enter to close this window.' _
fi
exit $status
