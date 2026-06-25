#!/bin/bash
# Run the AutoTTV pipeline on TOIs from toi_catalog_240226_for_ttv.csv by line number range.
#
# Usage:
#   ./run_batch_lines.sh <first_line> <last_line>
#   ./run_batch_lines.sh 2 10        # Run lines 2-10 (line 1 is the header)
#   ./run_batch_lines.sh 5 5         # Run only line 5
#
# The TOI is read from the fifth column of toi_catalog_240226_for_ttv.csv.
# Line 1 is the header row, so data starts at line 2.

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <first_line> <last_line>"
    echo "  first_line: first line number in toi_catalog_240226_for_ttv.csv (header is line 1)"
    echo "  last_line:  last line number in toi_catalog_240226_for_ttv.csv"
    exit 1
fi

FIRST=$1
LAST=$2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSV_FILE="$SCRIPT_DIR/toi_catalog_240226_for_ttv.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: $CSV_FILE not found"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$CSV_FILE" | tr -d ' ')

if [ "$FIRST" -lt 2 ]; then
    echo "Error: first_line must be >= 2 (line 1 is the header)"
    exit 1
fi

if [ "$LAST" -gt "$TOTAL_LINES" ]; then
    echo "Error: last_line ($LAST) exceeds file length ($TOTAL_LINES lines)"
    exit 1
fi

if [ "$FIRST" -gt "$LAST" ]; then
    echo "Error: first_line ($FIRST) > last_line ($LAST)"
    exit 1
fi

echo "Processing lines $FIRST to $LAST from $CSV_FILE"
echo "========================================="

SUCCESS=0
FAIL=0
SKIPPED=0

for LINE_NUM in $(seq "$FIRST" "$LAST"); do
    # Extract the fifth column (TOI) from the CSV line
    TOI=$(sed -n "${LINE_NUM}p" "$CSV_FILE" | cut -d',' -f5 | tr -d ' "')

    if [ -z "$TOI" ]; then
        echo "[$LINE_NUM] Skipping empty line"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo ""
    echo "[$LINE_NUM] Running pipeline for TOI $TOI ..."
    echo "-----------------------------------------"

    if python3 "$SCRIPT_DIR/run_full_analysis.py" "$TOI" --cpus=64; then
        echo "[$LINE_NUM] TOI $TOI completed successfully"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "[$LINE_NUM] TOI $TOI FAILED"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "========================================="
echo "Batch complete: $SUCCESS succeeded, $FAIL failed, $SKIPPED skipped"
echo "Total: $((SUCCESS + FAIL + SKIPPED)) of $((LAST - FIRST + 1)) lines"
