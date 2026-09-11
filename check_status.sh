#!/bin/bash

echo "=== Project Status Report ===" 
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check Task 1
echo "Task 1 (CRC-MSI): ✓ COMPLETED"
if [ -f "/storage/jmabq/Users/howl/First try/task1_output/metrics.json" ]; then
  echo "  Metrics:"
  cat "/storage/jmabq/Users/howl/First try/task1_output/metrics.json" | sed 's/^/    /'
fi
echo ""

# Check Task 2
echo "Task 2 (LUAD vs LUSC): RUNNING"
TASK2_LOG="/storage/jmabq/Users/howl/First try/task2_output/training.log"
if [ -f "$TASK2_LOG" ]; then
  LAST_EPOCH=$(grep "^Epoch" "$TASK2_LOG" | tail -1)
  echo "  $LAST_EPOCH"
fi
echo ""

# Check Task 3
echo "Task 3 (IM Detection): RUNNING"
TASK3_LOG="/storage/jmabq/Users/howl/First try/task3_output/training.log"
if [ -f "$TASK3_LOG" ]; then
  LAST_EPOCH=$(grep "^Epoch" "$TASK3_LOG" | tail -1)
  if [ -z "$LAST_EPOCH" ]; then
    echo "  Starting training..."
    grep "Training:" "$TASK3_LOG" | tail -1 | sed 's/^/  /'
  else
    echo "  $LAST_EPOCH"
  fi
fi
