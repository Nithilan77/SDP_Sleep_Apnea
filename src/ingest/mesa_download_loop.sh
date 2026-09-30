#!/bin/bash
# Re-runs the downloader until it finishes (DONE failed=0), backing off when the NSRR server is down.
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/share/gem/ruby/3.2.0/bin:$PATH"
for pass in $(seq 1 200); do
  ruby src/ingest/mesa_download.rb 2>&1 | sed -u "s/$NSRR_TOKEN/***/g" >> data/mesa/download.log
  tail -1 data/mesa/download.log | grep -q "DONE failed=0" && { echo "ALL DONE"; exit 0; }
  sleep 600
done
