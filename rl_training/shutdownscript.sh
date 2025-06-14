#!/bin/bash

MY_PROGRAM="mlagents-learn" # For example, "apache2" or "nginx"

echo "Shutting down!  Seeing if ${MY_PROGRAM} is running."

# Find the oldest copy of $MY_PROGRAM. Mlagents main instance always starts before the others.
PID="$(pgrep -o "$MY_PROGRAM")"

if [[ "$?" -ne 0 ]]; then
  echo "${MY_PROGRAM} not running, shutting down immediately."
  exit 0
fi

echo "Sending SIGINT to $PID"
kill -2 "$PID"

# Portable waitpid equivalent
while kill -0 "$PID"; do
   sleep 25
done

echo "$PID is done"