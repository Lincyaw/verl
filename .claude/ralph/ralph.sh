#!/bin/bash
set -e


if [ -z "$1" ]; then
  echo "Usage: $0 <max iterations>"
  echo "Example: $0 5"
  exit 1
fi

MAX_ITER=$1


echo "Starting Ralph loop, max iterations: $MAX_ITER"

for ((i=1; i<=MAX_ITER; i++)); do
  echo "========================================"
  echo "Iteration $i / $MAX_ITER"
  echo "========================================"

  result=$(ccr code --dangerously-skip-permissions -p "@.claude/ralph/plans/prd.json @.claude/ralph/progress.txt \
  1. Find the highest-priority feature to work on and work only on that feature. \
  This should be the one YOU decide has the highest priority - not necessarily the first item. \
  2. Update the PRD with the work that was done. \
  3. Append your progress to the progress.txt file. \
  Use this to leave a note for the next person working in the codebase. \
  4. Make a git commit of that feature. \
  5. Check CLAUDE.md to see if there are any special instructions for committing code, e.g., format code, lint, etc. \
  ONLY WORK ON A SINGLE FEATURE. \
  If, while implementing the feature, you notice the PRD is complete, output <promise>COMPLETE</promise> here.")

  echo "$result"

  if [[ "$result" == *"<promise>COMPLETE</promise>"* ]]; then
    echo "🎉 All tasks in the PRD are complete! Exiting loop."
    break
  fi

  echo "Iteration $i completed."
done