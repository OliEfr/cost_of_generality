#!/usr/bin/env bash
# Event watcher for the D31 T1/T3 training wave (24 cells). Emits one line per terminal state.
IDS="58103115 58103119 58103132 58103137 58103141 58103148 58103155 58103158 58103165 58103169 58103172 58103175 58103187 58103193 58103204 58103224 58103227 58103244 58103259 58103266 58103269 58103281 58103285 58103288"
seen=""
while true; do
  out=$(timeout 90 ssh leonardo "sacct -X -j $(echo $IDS | tr ' ' ',') -n -o JobID,JobName%20,State" 2>/dev/null) || { sleep 300; continue; }
  while read -r id name state rest; do
    [ -z "$id" ] && continue
    case "$state" in
      COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_ME*)
        echo "$seen" | grep -q " $id " || { echo "TRAIN $state $id $name"; seen="$seen $id "; } ;;
    esac
  done <<< "$out"
  n=$(echo $seen | wc -w)
  [ "$n" -ge 24 ] && { echo "TRAIN_WAVE_DONE 24/24 terminal"; break; }
  sleep 300
done
