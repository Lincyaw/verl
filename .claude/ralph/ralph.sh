#!/bin/bash
set -e


if [ -z "$1" ]; then
  echo "Usage: $0 <max iterations>"
  echo "Example: $0 5"
  exit 1
fi

MAX_ITER=$1

LOG_DIR=".claude/ralph/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/ralph_${TIMESTAMP}.log"

log() {
  echo "$1" | tee -a "$MAIN_LOG"
}

log "Starting Ralph loop, max iterations: $MAX_ITER"
log "Logs will be saved to: $MAIN_LOG"

PROMPT="@.claude/ralph/plans/prd.json @.claude/ralph/progress.txt  progress.txt may be long, so use tail, or grep to search it by id, like 'P0-BE-005'. \
  1. Find the highest-priority feature to work on and work only on that feature. \
  This should be the one YOU decide has the highest priority - not necessarily the first item. \
  2. If the prd.json is not clear, you can reference .claude/ralph/docs/spec.md and .claude/ralph/docs/task.md, but be careful the files is long, use grep, tail, head to find relevant information. \
  3. Update the PRD with the work that was done. \
  4. Append your progress to the progress.txt file, use this to leave a note for the next person working in the codebase. \
  5. If there are some questions you have about the design, implementation, etc., stop your work and ask me. \
  6. Call proper subagent and use proper skills for specfic tasks, but do not use Plan agent unless you feel it is must. \
  7. Make a git commit of that feature. \
  ONLY WORK ON A SINGLE FEATURE. \
  8. Check CLAUDE.md to see if there are any special instructions for committing code, e.g., format code, lint, etc. \
  If, while implementing the feature, you notice the PRD is complete, output <promise>COMPLETE</promise> here."

for ((i=1; i<=MAX_ITER; i++)); do
  log "========================================"
  log "Iteration $i / $MAX_ITER - $(date)"
  log "========================================"

  ITER_LOG="$LOG_DIR/iter_${TIMESTAMP}_${i}.log"
  ITER_JSON_LOG="$LOG_DIR/iter_${TIMESTAMP}_${i}.jsonl"
  
  TEMP_RESULT=$(mktemp)
  
  claude --dangerously-skip-permissions -p --verbose --output-format stream-json "$PROMPT" 2>&1 | \
    tee "$ITER_JSON_LOG" | \
    while IFS= read -r line; do
      echo "$line" >> "$ITER_LOG"
      if echo "$line" | jq -e 'select(.type == "assistant")' >/dev/null 2>&1; then
        echo "$line" | jq -r '.message.content[]? | select(.type == "text") | .text // empty' 2>/dev/null >> "$TEMP_RESULT"
      fi
    done
  
  result=$(cat "$TEMP_RESULT")
  rm -f "$TEMP_RESULT"

  echo "$result" >> "$MAIN_LOG"
  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    log "🎉 All tasks in the PRD are complete! Exiting loop."
    break
  fi

  log "Iteration $i completed."
done

log "========================================"
log "Ralph loop finished at $(date)"
log "Full log saved to: $MAIN_LOG"
log "Individual iteration logs in: $LOG_DIR/"
log "JSON stream logs: $LOG_DIR/iter_*.json"
