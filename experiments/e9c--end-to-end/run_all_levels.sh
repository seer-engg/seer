#!/bin/bash
# Run E9c experiment for all levels, multiple times
# Usage: ./run_all_levels.sh [num_iterations] [variant]
# Default: 2 iterations per level, variant 1
# Variants: 1=read-only (list issues), 2=simple write (create issue)

NUM_ITERATIONS=${1:-2}
VARIANT=${2:-1}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 E9c: Running all levels with $NUM_ITERATIONS iterations each"
echo "📋 Using test variant: $VARIANT"
echo "============================================================"

for level in 1 2 3; do
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Level $level - Running $NUM_ITERATIONS iterations (Variant $VARIANT)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for iter in $(seq 1 $NUM_ITERATIONS); do
        echo ""
        echo "  Iteration $iter/$NUM_ITERATIONS for Level $level"
        echo "  ────────────────────────────────────────────────"
        
        python3 e9c_end_to_end.py --level $level --variant $VARIANT
        
        if [ $? -ne 0 ]; then
            echo "  ❌ Iteration $iter failed for Level $level"
        else
            echo "  ✅ Iteration $iter completed for Level $level"
        fi
        
        # Small delay between iterations
        sleep 5
    done
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All runs completed!"
echo "Results saved in: results.json (consolidated format)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

